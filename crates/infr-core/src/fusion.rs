//! Shared peephole graph-rewrite fusion pass.
//!
//! Every GPU backend (Vulkan, Metal) independently re-implemented the SAME device-agnostic
//! peephole rewrites over the [`Graph`] IR — matching adjacent ops and folding them into one fused
//! kernel dispatch. This module hosts that logic ONCE; each backend supplies a [`FusionCfg`] naming
//! which patterns it can fuse and, per pattern, a `fn(DType) -> bool` predicate intersecting the
//! rewrite with the backend's own fused-kernel coverage. The result is a [`FusionPlan`] the backend
//! consumes exactly as it consumed its private `(fused, skip)` pair before.
//!
//! Two patterns are covered:
//!
//! 1. **`Linear(m==1) → Add(residual)`** ([`FusionCfg::linear_add`]) — fold a decode projection's
//!    following residual `Add` into the GEMV epilogue (`dst = gemv + residual`), killing the
//!    standalone `Add` kernel and the round-trip of the un-added projection. The decode
//!    `o_proj`/`down_proj` shape. Vulkan and Metal both do this.
//! 2. **`Rope/QkNormRope → WriteKv`** ([`FusionCfg::kv_write`]) — redirect a fused rope kernel's
//!    f16 K-row write straight into an f16 KV cache, absorbing the standalone `WriteKv`. Vulkan's
//!    `kv_write_peephole`, including the SWA ring-wrap guard.
//!
//! ## Live-range bounding (a correctness fix, applied to ALL backends)
//!
//! The live-range bound guards that the fused `dst`'s scratch handle is not consumed
//! by anything other than the absorbing op before it is next rewritten.
//! On the graphs the seam emits a fused Linear `dst` is
//! single-use, so it never un-fuses a real pair — it only prevents an unsafe fold. Keeping it
//! everywhere means the shared pass is correct for any future graph, not just today's.

use crate::graph::{Graph, Op, TensorKind};
use crate::tensor::{DType, TensorId};
use std::collections::{HashMap, HashSet};

/// Per-pattern config for [`FusionCfg::linear_add`].
pub struct LinearAddCfg<'a> {
    /// Fuse only when the `Linear` weight's dtype passes this predicate — the backend's
    /// fused-residual-GEMV kernel coverage (e.g. Vulkan `native_dense_supported || F16`, Metal's
    /// legacy+Q4K/Q6K list).
    pub weight_ok: &'a dyn Fn(DType) -> bool,
    /// `false` disables this pass entirely — the backend's escape hatch, taken off ITS config
    /// (Vulkan `kernels.vulkan.fuse_add` / `INFR_NO_FUSE_ADD`). A backend with no hatch passes `true`.
    pub enabled: bool,
}

/// Which peephole rewrites a backend wants planned, and the per-pattern predicates/hatches.
/// A `None`/`false` field leaves that pattern un-planned (its ops stay split).
pub struct FusionCfg<'a> {
    /// `Linear(m==1) → Add` residual fusion.
    pub linear_add: Option<LinearAddCfg<'a>>,
    /// `Rope/QkNormRope → WriteKv` fusion (f16 cache only; predicate is fixed f16, so a plain flag).
    pub kv_write: bool,
}

/// The planned rewrites for one graph. Each map is keyed by the op index of the op that ABSORBS
/// its neighbour; `skip` holds every op index the executor must NOT dispatch (the absorbed ops).
#[derive(Default)]
pub struct FusionPlan {
    /// `Linear` op idx → (residual operand, final `Add` dst). The absorbed `Add` is at `idx + 1`.
    pub linear_add: HashMap<usize, (TensorId, TensorId)>,
    /// `Rope`/`QkNormRope` op idx → (KV cache tensor, write row). The absorbed `WriteKv` is at
    /// `idx + 1`.
    pub kv_write: HashMap<usize, (TensorId, usize)>,
    /// Op indices the executor elides (absorbed `Add`s and absorbed `WriteKv`s).
    pub skip: HashSet<usize>,
}

/// Plan the peephole fusions `cfg` enables over `graph`. Pure host logic over the IR — no device
/// types, and (since the config campaign) no environment: each pattern's escape hatch arrives
/// as the `enabled` flag its backend read off its own `Config`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn plan_fusions(graph: &Graph, cfg: &FusionCfg) -> FusionPlan {
    let mut plan = FusionPlan::default();
    if cfg.kv_write {
        plan_kv_write(graph, &mut plan);
    }
    if let Some(c) = cfg.linear_add.as_ref().filter(|c| c.enabled) {
        plan_linear_add(graph, c, &mut plan);
    }
    plan
}

/// `Linear(m==1, Internal dst, covered weight) → Add(residual)`: fold the following residual `Add`
/// into the GEMV epilogue. Only the IMMEDIATELY following op fuses (the seam emits the pair
/// adjacent for non-gemma models; gemma's sandwich norm sits between and correctly blocks it).
fn plan_linear_add(graph: &Graph, cfg: &LinearAddCfg, plan: &mut FusionPlan) {
    for (i, op) in graph.ops.iter().enumerate() {
        let Op::Linear {
            dst,
            m: 1,
            weight,
            out_f,
            ..
        } = op
        else {
            continue;
        };
        if !matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal) {
            continue;
        }
        if !(cfg.weight_ok)(graph.desc(*weight).dtype) {
            continue;
        }
        let Some(Op::Add {
            a,
            b,
            dst: add_dst,
            n,
        }) = graph.ops.get(i + 1)
        else {
            continue;
        };
        if *n != *out_f {
            continue;
        }
        let residual = if b == dst && a != dst {
            *a
        } else if a == dst && b != dst {
            *b
        } else {
            continue;
        };
        // Live-range bound (see module docs): the Linear's `dst` must be consumed ONLY by this Add
        // before it is next rewritten — eliding the standalone write is unsafe if anything else
        // reads the un-added projection (the `dst` scratch may be recycled by a later layer).
        if !dst_only_read_by_next(graph, i + 2, *dst) {
            continue;
        }
        plan.linear_add.insert(i, (residual, *add_dst));
        plan.skip.insert(i + 1);
    }
}

/// `Rope`/`QkNormRope(Internal f16 dst) → WriteKv(f16 cache)`: redirect the fused rope kernel's f16
/// K-row write straight into the KV cache, absorbing the standalone `WriteKv`.
fn plan_kv_write(graph: &Graph, plan: &mut FusionPlan) {
    for (i, op) in graph.ops.iter().enumerate() {
        // QkNormRope (qwen/gemma) or f16-out Rope (llama) — both write the f16 K row the peephole
        // can redirect straight into the KV cache.
        let kxx = match op {
            Op::QkNormRope { dst, .. } | Op::Rope { dst, .. } => dst,
            _ => continue,
        };
        // Only fuse an Internal (scratch) dst (we redirect the write into the KV cache). The output
        // must be f16 (the shader casts f32→f16); WriteKv of an f16 src is a plain copy.
        if !matches!(graph.tensors[kxx.0 as usize].kind, TensorKind::Internal) {
            continue;
        }
        if !matches!(graph.desc(*kxx).dtype, DType::F16) {
            continue;
        }
        let Some(Op::WriteKv {
            src,
            cache,
            pos,
            rows,
            row_stride,
        }) = graph.ops.get(i + 1)
        else {
            continue;
        };
        // A Q8_0 cache needs a real quantizing WriteKv (store_q8), so DON'T fuse the f16 rope write
        // into it — leave the standalone WriteKv to run.
        if src == kxx && matches!(graph.desc(*cache).dtype, DType::F16) {
            // SWA ring cache (row capacity < full context): the write row is pos % cap_rows. The
            // fused rope kernels write `rows` CONTIGUOUS rows from out_base, so a batched prefill
            // write that would cross the ring's wrap boundary can't fuse — leave the standalone
            // WriteKv, whose lowering splits the write at the wrap. Decode (rows == 1) always fuses;
            // a full-context cache never wraps (pos < cap_rows).
            let cap_rows = graph.desc(*cache).numel() / (*row_stride as usize).max(1);
            let pos_r = if cap_rows > 0 {
                *pos as usize % cap_rows
            } else {
                *pos as usize
            };
            if cap_rows == 0 || pos_r + *rows as usize <= cap_rows {
                plan.kv_write.insert(i, (*cache, pos_r));
                plan.skip.insert(i + 1);
            }
        }
    }
}

/// Live-range check: from op index `start`, is `dst` read by NOTHING before it is next rewritten?
/// Returns `true` if `dst` is dead until its next write (or graph end) — the fold is safe.
///
/// `pub` because a backend post-filtering [`FusionPlan::kv_write`] needs the SAME bound: that plan
/// carries no live-range guard (Vulkan's record-once decode path REQUIRES its K write fused, so it
/// cannot be un-fused by a graph-shape test), and a backend that redirects the rope's write into the
/// cache leaves the rope's `dst` scratch unwritten — safe only if nothing reads it. Sharing this
/// function keeps that check the same one every other pattern uses.
pub fn dst_only_read_by_next(graph: &Graph, start: usize, dst: TensorId) -> bool {
    for j in start..graph.ops.len() {
        let (ins, outs) = graph.ops[j].io();
        if ins.contains(&dst) {
            return false;
        }
        if outs.contains(&dst) {
            break;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::graph::Op;
    use crate::tensor::TensorDesc;

    /// A minimal decode `Linear(m==1) -> Add(residual)` pair, the shape both `linear_add` backends
    /// fuse.
    fn linear_add_graph() -> Graph {
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![1, 4], DType::F32));
        let w = g.weight(TensorDesc::new(vec![4, 4], DType::F16));
        let dst = g.internal(TensorDesc::new(vec![1, 4], DType::F32));
        let res = g.input(TensorDesc::new(vec![1, 4], DType::F32));
        let out = g.output(TensorDesc::new(vec![1, 4], DType::F32));
        g.push(Op::Linear {
            x,
            weight: w,
            dst,
            m: 1,
            in_f: 4,
            out_f: 4,
            w_off: 0,
        });
        g.push(Op::Add {
            a: dst,
            b: res,
            dst: out,
            n: 4,
        });
        g
    }

    /// The escape hatch, driven through a `Config` rather than the process environment: the
    /// backend hands `plan_fusions` the POSITIVE `fuse_add` field, and a cleared field leaves the
    /// pair split (both ops dispatched) exactly as `INFR_NO_FUSE_ADD` used to.
    #[test]
    fn linear_add_hatch_comes_from_the_config() {
        let g = linear_add_graph();
        let all: &dyn Fn(DType) -> bool = &|_| true;
        let plan_with = |enabled: bool| {
            plan_fusions(
                &g,
                &FusionCfg {
                    linear_add: Some(LinearAddCfg {
                        weight_ok: all,
                        enabled,
                    }),
                    kv_write: false,
                },
            )
        };

        // Shipped default: the fold is planned and the standalone `Add` is elided.
        let d = Config::default();
        assert!(d.kernels.vulkan.fuse_add);
        let on = plan_with(d.kernels.vulkan.fuse_add);
        assert_eq!(on.linear_add.len(), 1);
        assert!(on.skip.contains(&1));

        // The `NO_FUSE` keys clear their POSITIVE field on PRESENCE — value irrelevant,
        // including `=0` (the presence-inv truth table).
        for raw in ["", "0", "1"] {
            let layer =
                crate::config::env::parse(&|k| (k == "INFR_NO_FUSE_ADD").then(|| raw.to_string()))
                    .expect("INFR_NO_FUSE_ADD");
            let cfg = Config::load_from_layers(&[layer]);
            assert!(
                !cfg.kernels.vulkan.fuse_add,
                "INFR_NO_FUSE_ADD={raw:?} must clear its fusion field"
            );
        }

        // Cleared field => nothing planned, nothing skipped: both ops dispatch.
        let off = plan_with(false);
        assert!(off.linear_add.is_empty());
        assert!(off.skip.is_empty());
    }
}
