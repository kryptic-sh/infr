//! The routing half of the teacher-forced comparison (`crate::tolerance`, `docs/gpu-tolerance.md`)
//! for a routed-expert (MoE) model: what a run's routers selected ([`Routing`]), and what the
//! runner adds to a graph to read a selection back and to run the experts on a supplied one
//! ([`RouteTap`]). Nothing here is reached outside a forced run (`super::forced_run`).
use crate::Config;
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};

/// The expert selections of a routed-expert (MoE) model's routers over one teacher-forced run:
/// for every position fed (the prompt's and each forced token's, from 0) and every
/// router-selected layer, the experts the backend's own router picked there. On a run forced
/// along another run's selection these are still the run's OWN picks, made on hidden states
/// computed under the forced selection: what it would have run had it been left to itself.
#[derive(Clone, Debug, PartialEq)]
pub struct Routing {
    /// The router-selected layers, ascending (a hash-routed layer selects by token, not by
    /// router, and is not one).
    pub layers: Vec<usize>,
    /// Experts a router selects per position.
    pub n_used: usize,
    pub n_expert: usize,
    /// `[positions * layers.len() * n_used]`: each selection, best first.
    pub ids: Vec<u32>,
    /// `[positions * layers.len() * n_expert]`: the score each expert was ranked by (its gated
    /// probability plus the selection bias; negative infinity in an expert group that lost).
    /// Empty where the backend's router does not produce them (Vulkan).
    pub scores: Vec<f32>,
}

impl Routing {
    /// How many positions the run fed.
    pub fn positions(&self) -> usize {
        self.ids.len() / (self.layers.len() * self.n_used)
    }

    /// Where `(pos, layer)` sits among the (position, layer) pairs, if the run has it.
    fn slot(&self, pos: usize, layer: usize) -> Option<usize> {
        let l = self.layers.iter().position(|&x| x == layer)?;
        (pos < self.positions()).then_some(pos * self.layers.len() + l)
    }

    /// The experts selected at `pos` in `layer`, best first.
    pub fn ids_at(&self, pos: usize, layer: usize) -> Option<&[u32]> {
        let i = self.slot(pos, layer)? * self.n_used;
        Some(&self.ids[i..i + self.n_used])
    }

    /// Every expert's selection score at `pos` in `layer`, when the backend gave them.
    pub fn scores_at(&self, pos: usize, layer: usize) -> Option<&[f32]> {
        let i = self.slot(pos, layer)? * self.n_expert;
        self.scores.get(i..i + self.n_expert)
    }
}

/// The [`Routing`] of a run that fed `positions` positions through the router-selected `layers`,
/// from what its graphs read back; an error unless every (position, layer) was read back exactly
/// once and each readback is a selection: `n_used` distinct experts.
pub(super) fn assemble_routing(
    layers: Vec<usize>,
    mc: &crate::kv::MoeConfig,
    taken: &[RouteTaken],
    positions: usize,
    with_scores: bool,
) -> AResult<Routing> {
    let pairs = positions * layers.len();
    let mut r = Routing {
        n_used: mc.n_used,
        n_expert: mc.n_expert,
        ids: vec![0; pairs * mc.n_used],
        scores: vec![0.0; if with_scores { pairs * mc.n_expert } else { 0 }],
        layers,
    };
    let mut seen = vec![false; pairs];
    for t in taken {
        for row in 0..t.rows {
            let at = r.slot(t.pos + row, t.layer).ok_or_else(|| {
                anyhow!(
                    "a selection was read back for position {}, layer {}, outside the run",
                    t.pos + row,
                    t.layer
                )
            })?;
            if std::mem::replace(&mut seen[at], true) {
                return Err(anyhow!(
                    "position {}, layer {} was routed twice",
                    t.pos + row,
                    t.layer
                ));
            }
            let (k, e) = (r.n_used, r.n_expert);
            r.ids[at * k..(at + 1) * k].copy_from_slice(&t.ids[row * k..(row + 1) * k]);
            if with_scores {
                r.scores[at * e..(at + 1) * e].copy_from_slice(&t.scores[row * e..(row + 1) * e]);
            }
        }
    }
    if let Some(at) = seen.iter().position(|s| !s) {
        return Err(anyhow!(
            "no selection was read back for position {}, layer {}",
            at / r.layers.len(),
            r.layers[at % r.layers.len()]
        ));
    }
    // A buffer the selection kernel never wrote, or a top-k that picked one expert twice, reads
    // back as ids all the same; neither is a selection.
    for (at, sel) in r.ids.chunks(r.n_used).enumerate() {
        let (pos, layer) = (at / r.layers.len(), r.layers[at % r.layers.len()]);
        if let Some(&e) = sel.iter().find(|&&e| e as usize >= r.n_expert) {
            return Err(anyhow!(
                "the selection read back at position {pos}, layer {layer} names expert {e} of {}",
                r.n_expert
            ));
        }
        if let Some((i, &e)) = sel.iter().enumerate().find(|&(i, e)| sel[..i].contains(e)) {
            return Err(anyhow!(
                "the selection read back at position {pos}, layer {layer} names expert {e} twice \
                 (slot {i}): {sel:?}"
            ));
        }
    }
    Ok(r)
}

/// What the teacher-forced harness asks of the model's router-selected expert layers (see
/// `generate_dense_backend_bound`'s `route`), and what it got back.
pub(crate) struct RouteTap<'a> {
    /// The selection every such layer runs on, at every position this call feeds: a reference
    /// run's. `None` leaves each layer on its own router's.
    pub(crate) forced: Option<&'a Routing>,
    /// Read back the scores each selection was ranked by, too (`Op::MoeRoute::dst_scores`).
    pub(crate) scores: bool,
    /// The backend's own selections, one entry per executed graph and layer, in execution order.
    pub(crate) taken: Vec<RouteTaken>,
}

/// The router's own selection at one layer for `rows` consecutive positions from `pos`.
pub(crate) struct RouteTaken {
    pub(crate) pos: usize,
    pub(crate) rows: usize,
    pub(crate) layer: usize,
    /// `[rows * n_used]`, best first.
    pub(crate) ids: Vec<u32>,
    /// `[rows * n_expert]`; empty unless [`RouteTap::scores`].
    pub(crate) scores: Vec<f32>,
}

/// What a build adds for a [`RouteTap`]: the readback, and whether the layers run on a supplied
/// selection.
#[derive(Clone, Copy)]
pub(super) struct RouteAsk {
    pub(super) forced: bool,
    pub(super) scores: bool,
}

/// One layer's graph handles for a [`RouteTap`].
pub(super) struct RouteHandles {
    pub(super) layer: usize,
    own: TensorId,
    scores: Option<TensorId>,
    forced: Option<TensorId>,
}

/// The layers whose experts the router selects, which a [`RouteTap`] covers: every routed layer
/// but the hash-routed ones (deepseek4), whose selection is the token's table row.
pub(crate) fn router_selected_layers(c: &Config) -> Vec<usize> {
    (0..c.n_layer)
        .filter(|&l| {
            c.dual_moe() || (c.moe.is_some() && c.is_moe_layer(l) && !c.is_hash_moe_layer(l))
        })
        .collect()
}

/// For a [`RouteTap`]: emit the readback of the router's own selection at `layer`
/// (`Op::MoeRoute`, on the operands the layer's `Op::MoeFfn` routes with) and, when the run is
/// forced along a reference's routing, declare the input that op takes its selection from
/// instead. Returns that input.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_route(
    g: &mut Graph,
    ask: RouteAsk,
    out: &mut Vec<RouteHandles>,
    layer: usize,
    batch: usize,
    ne: usize,
    mc: &crate::kv::MoeConfig,
    router_x: TensorId,
    router: TensorId,
    exp_probs_b: Option<TensorId>,
) -> Option<TensorId> {
    let own = g.output(TensorDesc::new(vec![batch, mc.n_used], DType::F32));
    let scores = ask
        .scores
        .then(|| g.output(TensorDesc::new(vec![batch, mc.n_expert], DType::F32)));
    g.push(Op::MoeRoute {
        router_x,
        router,
        exp_probs_b,
        dst_ids: own,
        dst_scores: scores,
        ne: ne as u32,
        n_expert: mc.n_expert as u32,
        n_used: mc.n_used as u32,
        gating: mc.gating,
        n_expert_groups: mc.n_expert_groups,
        n_expert_groups_used: mc.n_expert_groups_used,
    });
    let forced = ask
        .forced
        .then(|| g.input(TensorDesc::new(vec![batch, mc.n_used], DType::I32)));
    out.push(RouteHandles {
        layer,
        own,
        scores,
        forced,
    });
    forced
}

/// The buffers one graph's [`RouteHandles`] bind, for `rows` positions.
pub(super) struct RouteBufs {
    layers: Vec<usize>,
    rows: usize,
    own: Vec<Box<dyn Buffer>>,
    scores: Vec<Option<Box<dyn Buffer>>>,
    forced: Vec<Option<Box<dyn Buffer>>>,
}

impl RouteBufs {
    pub(super) fn alloc(
        be: &dyn Backend,
        layers: Vec<usize>,
        rows: usize,
        mc: &crate::kv::MoeConfig,
        tap: &RouteTap,
    ) -> AResult<Self> {
        let alloc = |n: usize, usage| be.alloc(n * 4, usage).map_err(|e| anyhow!("{e}"));
        let mut b = RouteBufs {
            rows,
            own: Vec::new(),
            scores: Vec::new(),
            forced: Vec::new(),
            layers,
        };
        for _ in &b.layers {
            b.own.push(alloc(rows * mc.n_used, BufferUsage::Readback)?);
            b.scores.push(match tap.scores {
                true => Some(alloc(rows * mc.n_expert, BufferUsage::Readback)?),
                false => None,
            });
            b.forced.push(match tap.forced {
                Some(_) => Some(alloc(rows * mc.n_used, BufferUsage::Staging)?),
                None => None,
            });
        }
        Ok(b)
    }

    /// Bind every one of a graph's handles to its buffer. The two were made from the same
    /// layers, or the graph is not the one these buffers were sized for.
    pub(super) fn bind<'b>(&'b self, b: &mut Bindings<'b>, handles: &[RouteHandles]) {
        assert!(
            handles
                .iter()
                .map(|r| r.layer)
                .eq(self.layers.iter().copied()),
            "the graph's router-selected layers are not the ones its routing buffers were made for"
        );
        for (i, r) in handles.iter().enumerate() {
            b.bind(r.own, self.own[i].as_ref());
            if let (Some(id), Some(buf)) = (r.scores, &self.scores[i]) {
                b.bind(id, buf.as_ref());
            }
            if let (Some(id), Some(buf)) = (r.forced, &self.forced[i]) {
                b.bind(id, buf.as_ref());
            }
        }
    }

    /// Upload the reference's selection for the positions from `pos`, when the run is forced.
    pub(super) fn upload(&self, be: &dyn Backend, tap: &RouteTap, pos: usize) -> AResult<()> {
        let Some(want) = tap.forced else {
            return Ok(());
        };
        for (i, &layer) in self.layers.iter().enumerate() {
            let mut ids: Vec<i32> = Vec::with_capacity(self.rows * want.n_used);
            for p in pos..pos + self.rows {
                let row = want.ids_at(p, layer).ok_or_else(|| {
                    anyhow!(
                        "the reference routing has no selection for position {p}, layer {layer}"
                    )
                })?;
                ids.extend(row.iter().map(|&e| e as i32));
            }
            let buf = self.forced[i]
                .as_ref()
                .expect("a forced run allocates the selection inputs");
            be.upload(buf.as_ref(), bytemuck::cast_slice(&ids))
                .map_err(|e| anyhow!("{e}"))?;
        }
        Ok(())
    }

    /// Read back what the graph just executed for the positions from `pos` selected on its own.
    pub(super) fn download(&self, be: &dyn Backend, tap: &mut RouteTap, pos: usize) -> AResult<()> {
        for (i, &layer) in self.layers.iter().enumerate() {
            let mut ids = vec![0u32; self.own[i].len_bytes() / 4];
            be.download(self.own[i].as_ref(), bytemuck::cast_slice_mut(&mut ids))
                .map_err(|e| anyhow!("{e}"))?;
            let mut scores = Vec::new();
            if let Some(buf) = &self.scores[i] {
                scores.resize(buf.len_bytes() / 4, 0f32);
                be.download(buf.as_ref(), bytemuck::cast_slice_mut(&mut scores))
                    .map_err(|e| anyhow!("{e}"))?;
            }
            tap.taken.push(RouteTaken {
                pos,
                rows: self.rows,
                layer,
                ids,
                scores,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MC: crate::kv::MoeConfig = crate::kv::MoeConfig {
        n_expert: 4,
        n_used: 2,
        n_ff_exp: 8,
        scale: 1.0,
        gating: infr_core::graph::MoeGating::Softmax,
        norm_w: true,
        weight_before: false,
        n_expert_groups: 0,
        n_expert_groups_used: 0,
    };

    /// One layer's readback of `ids` for the positions from `pos`.
    fn taken(pos: usize, layer: usize, ids: &[u32]) -> RouteTaken {
        RouteTaken {
            pos,
            rows: ids.len() / MC.n_used,
            layer,
            ids: ids.to_vec(),
            scores: Vec::new(),
        }
    }

    #[test]
    fn a_complete_readback_assembles_by_position_and_layer() {
        // Two layers over three positions, read back out of position order and in two graphs.
        let got = [
            taken(1, 5, &[2, 3, 0, 3]),
            taken(1, 7, &[1, 0, 2, 1]),
            taken(0, 5, &[0, 1]),
            taken(0, 7, &[3, 2]),
        ];
        let r = assemble_routing(vec![5, 7], &MC, &got, 3, false).expect("complete");
        assert_eq!(r.positions(), 3);
        assert_eq!(r.ids_at(0, 5), Some(&[0, 1][..]));
        assert_eq!(r.ids_at(2, 5), Some(&[0, 3][..]));
        assert_eq!(r.ids_at(2, 7), Some(&[2, 1][..]));
        assert_eq!(r.ids_at(3, 5), None);
        assert_eq!(r.ids_at(0, 6), None);
        assert_eq!(r.scores_at(0, 5), None);
    }

    /// What is not a selection is refused by name: an expert named twice (a top-k that did not
    /// mask its pick), a buffer nothing wrote (all zeros), an id past the experts, a (position,
    /// layer) read back twice or never.
    #[test]
    fn a_readback_that_is_not_a_selection_is_refused() {
        let err = |got: &[RouteTaken], positions| {
            assemble_routing(vec![5], &MC, got, positions, false)
                .expect_err("refused")
                .to_string()
        };
        let e = err(&[taken(0, 5, &[0, 1, 2, 2])], 2);
        assert!(
            e.contains("position 1, layer 5 names expert 2 twice"),
            "{e}"
        );
        let e = err(&[taken(0, 5, &[0, 0, 0, 0])], 2);
        assert!(
            e.contains("position 0, layer 5 names expert 0 twice"),
            "{e}"
        );
        let e = err(&[taken(0, 5, &[0, 4])], 1);
        assert!(e.contains("names expert 4 of 4"), "{e}");
        let e = err(&[taken(0, 5, &[0, 1]), taken(0, 5, &[0, 1])], 1);
        assert!(e.contains("routed twice"), "{e}");
        let e = err(&[taken(0, 5, &[0, 1])], 2);
        assert!(
            e.contains("no selection was read back for position 1"),
            "{e}"
        );
        let e = err(&[taken(1, 5, &[0, 1])], 1);
        assert!(e.contains("outside the run"), "{e}");
    }
}
