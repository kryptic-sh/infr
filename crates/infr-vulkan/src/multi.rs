//! What the multi-device wrappers ([`crate::tp`], [`crate::ep`], [`crate::PipelineBackend`]) share:
//! each presents several [`VulkanBackend`]s as one [`Backend`] and runs one graph as a sequence of
//! segments, each compiled and executed as its own plan.

use std::collections::HashMap;

use infr_core::backend::{Backend, Capabilities};
use infr_core::error::Result;
use infr_core::graph::{Graph, TensorKind};
use infr_core::tensor::TensorId;

use crate::{be, VulkanBackend};

/// Refuse a segmentation of `graph` in which an `Internal` tensor is written in one segment and read
/// in a later one before that segment writes it again.
///
/// Each segment runs as its own plan, and a plan's `Internal` tensors are scratch the backend
/// allocates zero-filled per execute, so such a read sees zeros where the graph meant the earlier
/// value — a wrong answer with no error. A wrapper that does carry a tensor across a cut binds it or
/// promotes it out of `Internal` before calling this, so what is refused is exactly what nothing
/// carries.
///
/// `ends` are the segments' exclusive end op indices in ascending order (the last one is
/// `graph.ops.len()`); `who` names the wrapper and the knob that selected it, for the message.
pub(crate) fn refuse_scratch_across_cuts(graph: &Graph, ends: &[usize], who: &str) -> Result<()> {
    // Last writing op of each tensor written in an EARLIER segment than the current one.
    let mut earlier: HashMap<TensorId, usize> = HashMap::new();
    let mut lo = 0usize;
    for &hi in ends {
        let mut here: HashMap<TensorId, usize> = HashMap::new();
        for (i, op) in graph.ops.iter().enumerate().take(hi).skip(lo) {
            let (reads, writes) = op.io();
            for t in reads {
                let decl = &graph.tensors[t.0 as usize];
                if decl.kind != TensorKind::Internal || here.contains_key(&t) {
                    continue;
                }
                if let Some(&w) = earlier.get(&t) {
                    let label = decl
                        .label
                        .as_deref()
                        .map(|l| format!(" ({l})"))
                        .unwrap_or_default();
                    return Err(be(format!(
                        "{who}: op {i} ({}) reads scratch tensor {}{label}, which op {w} wrote \
                         before a cut that does not carry it — each side of the cut runs as its \
                         own plan with fresh zero-filled scratch, so the read would see zeros. \
                         This graph is not supported by {who}",
                        op.kind(),
                        t.0,
                    )));
                }
            }
            for t in writes {
                here.insert(t, i);
            }
        }
        earlier.extend(here);
        lo = hi;
    }
    Ok(())
}

/// The capabilities a multi-device wrapper reports for the work the runner sizes from them: the
/// MOST conservative value over every rank. A world mixing a discrete card with an integrated GPU
/// must size its prefill chunk for the integrated one (whose hang watchdog the smaller chunk
/// exists for), whichever of them is rank 0. Everything else is rank 0's, the per-device kernel
/// choices being each rank's own at compile.
pub(crate) fn conservative_capabilities(ranks: &[VulkanBackend]) -> Capabilities {
    merge_conservative(ranks.iter().map(|b| b.capabilities()))
}

/// [`conservative_capabilities`] over plain values, so the merge is testable without opening
/// several devices. Panics on an empty iterator (every wrapper refuses an empty world at `new`).
fn merge_conservative(mut caps: impl Iterator<Item = Capabilities>) -> Capabilities {
    let mut c = caps
        .next()
        .expect("a multi-device backend has at least one rank");
    for o in caps {
        c.integrated |= o.integrated;
        // 0 = unknown, which `infr_core::integrated_ubatch_rows` treats as the smallest part, so
        // the plain minimum is also the conservative choice.
        c.compute_units = c.compute_units.min(o.compute_units);
        c.max_shared_memory_bytes = c.max_shared_memory_bytes.min(o.max_shared_memory_bytes);
        c.unified_memory &= o.unified_memory;
        c.kv_swa_ring &= o.kv_swa_ring;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::graph::Op;
    use infr_core::tensor::{DType, TensorDesc};

    fn caps(integrated: bool, cu: u32, shared: u32) -> Capabilities {
        Capabilities {
            integrated,
            compute_units: cu,
            max_shared_memory_bytes: shared,
            unified_memory: integrated,
            kv_swa_ring: true,
            ..Default::default()
        }
    }

    /// A discrete rank 0 must not hide an integrated rank 1: the chunk the runner derives from the
    /// merged class is the one that keeps the integrated device inside its watchdog.
    #[test]
    fn merged_capabilities_take_the_weakest_rank() {
        let discrete = caps(false, 96, 65536);
        let igpu = caps(true, 2, 32768);
        for order in [[discrete.clone(), igpu.clone()], [igpu, discrete.clone()]] {
            let m = merge_conservative(order.into_iter());
            assert!(
                m.integrated,
                "an integrated rank makes the world integrated"
            );
            assert_eq!(m.compute_units, 2);
            assert_eq!(m.max_shared_memory_bytes, 32768);
            assert!(!m.unified_memory, "budget as the discrete rank does");
        }
        let one = merge_conservative(std::iter::once(discrete));
        assert!(!one.integrated);
        assert_eq!(one.compute_units, 96);
    }

    /// `x → [a = scale(x)] | cut | y = a + x`: `a` is scratch written before the cut and read after
    /// it. Promoted out of `Internal` (what a carrying wrapper does) it is accepted; a cut that
    /// falls elsewhere, or a later segment that rewrites `a` before reading it, is fine too.
    #[test]
    fn scratch_read_across_a_cut_is_refused() {
        let mut g = Graph::new();
        let d = || TensorDesc::new(vec![4], DType::F32);
        let x = g.input(d());
        let a = g.internal(d());
        let y = g.output(d());
        g.push(Op::Scale {
            x,
            dst: a,
            s: 2.0,
            n: 4,
        });
        g.push(Op::Add {
            a,
            b: x,
            dst: y,
            n: 4,
        });
        let err = refuse_scratch_across_cuts(&g, &[1, 2], "TEST_KNOB").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("TEST_KNOB"), "{msg}");
        assert!(msg.contains(&format!("tensor {}", a.0)), "{msg}");
        assert!(msg.contains("op 1 (Add)"), "{msg}");

        refuse_scratch_across_cuts(&g, &[2], "TEST_KNOB").expect("no cut, nothing crosses");

        let mut carried = g.clone();
        carried.tensors[a.0 as usize].kind = TensorKind::Input;
        refuse_scratch_across_cuts(&carried, &[1, 2], "TEST_KNOB").expect("a carried tensor");

        // The later segment recomputes `a` before reading it: nothing crosses.
        let mut rewritten = g.clone();
        rewritten.ops.insert(
            1,
            Op::Scale {
                x,
                dst: a,
                s: 3.0,
                n: 4,
            },
        );
        refuse_scratch_across_cuts(&rewritten, &[1, 3], "TEST_KNOB").expect("rewritten first");
    }

    /// Every wrapper reports an integrated world when any rank is integrated, with the discrete
    /// card as rank 0 — the order in which rank 0's own class would hide the integrated one. Opens
    /// the devices only; nothing runs on them.
    #[test]
    #[ignore = "requires a discrete and an integrated Vulkan GPU"]
    fn wrappers_report_an_integrated_rank_behind_a_discrete_rank_0() {
        let Ok(devs) = VulkanBackend::enumerate_devices_from_env() else {
            return; // no Vulkan — self-skip
        };
        let open = |i: usize| VulkanBackend::new_on(i).expect("open device");
        let class = |i: usize| open(i).capabilities().integrated;
        let discrete = devs.iter().map(|d| d.index).find(|&i| !class(i));
        let integrated = devs.iter().map(|d| d.index).find(|&i| class(i));
        let (Some(d), Some(ig)) = (discrete, integrated) else {
            eprintln!("skip: needs one discrete and one integrated Vulkan device");
            return;
        };
        let world = || vec![open(d), open(ig)];
        let want_cu = world().iter().map(|b| b.capabilities().compute_units).min();
        let tp = crate::TensorParallelBackend::new(world(), 2, 2, 2, false).expect("tp");
        let ep = crate::ExpertParallelBackend::new(world(), 2, false).expect("ep");
        let pipe = crate::PipelineBackend::new(world(), vec![0, 1], false).expect("pipeline");
        for (name, caps) in [
            ("tensor-parallel", tp.capabilities()),
            ("expert-parallel", ep.capabilities()),
            ("pipeline", pipe.capabilities()),
        ] {
            assert!(
                caps.integrated,
                "{name} hid its integrated rank behind rank 0"
            );
            assert_eq!(Some(caps.compute_units), want_cu, "{name}");
        }
    }
}
