//! Vulkan adapter for the agnostic `infr_core::Backend` seam: lower a `Graph` of composite ops onto
//! the existing fused Recorder kernels, recorded into one command buffer. Mirrors the CPU
//! interpreter (`infr-llama::cpu_backend`) op-for-op, but executes on the GPU — so the SAME model
//! `Graph` runs on either backend. Built incrementally; ops not yet mapped return an error.

use crate::{be, VulkanBackend};
use infr_core::backend::{Bindings, Buffer, BufferUsage, Plan};
use infr_core::error::Result;
use infr_core::graph::{Graph, Op, TensorKind};
use infr_core::{Backend, TensorId};

/// Compiled plan: the `Graph` replayed each `execute` (buffers re-bound per step, no recompile). A
/// later pass can pre-record a resubmittable command buffer (record-once decode) keyed by shape.
pub struct VkPlan {
    graph: Graph,
}

impl Plan for VkPlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(crate) fn compile(graph: &Graph) -> Result<Box<dyn Plan>> {
    Ok(Box::new(VkPlan {
        graph: graph.clone(),
    }))
}

/// Resolve a graph tensor to its device buffer: `Internal` from the per-execute scratch, everything
/// else (`Input`/`Weight`/`Output`) from the model-provided `Bindings`.
fn resolve<'a>(
    scratch: &'a [Option<Box<dyn Buffer>>],
    bindings: &'a Bindings,
    id: TensorId,
) -> Result<&'a dyn Buffer> {
    match &scratch[id.0 as usize] {
        Some(b) => Ok(b.as_ref()),
        None => bindings
            .get(id)
            .ok_or_else(|| be(format!("vulkan adapter: unbound tensor {}", id.0))),
    }
}

pub(crate) fn execute(be_: &VulkanBackend, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
    let graph = &plan
        .as_any()
        .downcast_ref::<VkPlan>()
        .ok_or_else(|| be("vulkan adapter: plan is not a VkPlan"))?
        .graph;

    // The model binds Input/Weight/Output to device buffers; the backend allocates only the
    // `Internal` scratch (activations), live for this one execute.
    let mut scratch: Vec<Option<Box<dyn Buffer>>> =
        (0..graph.tensors.len()).map(|_| None).collect();
    for (i, decl) in graph.tensors.iter().enumerate() {
        if matches!(decl.kind, TensorKind::Internal) {
            let bytes = decl
                .desc
                .dtype
                .dense_bytes(decl.desc.numel())
                .ok_or_else(|| be("vulkan adapter: internal tensor must be a dense dtype"))?;
            scratch[i] = Some(be_.alloc(bytes.max(4), BufferUsage::Activations)?);
        }
    }
    let r = |id: TensorId| resolve(&scratch, bindings, id);

    let rec = be_.recorder()?;
    for op in &graph.ops {
        match op {
            Op::RmsNorm {
                x,
                weight,
                dst,
                rows,
                dim,
                eps,
            } => {
                rec.rmsnorm(
                    r(*x)?,
                    r(*weight)?,
                    r(*dst)?,
                    *rows as usize,
                    *dim as usize,
                    *eps,
                );
            }
            other => {
                return Err(be(format!(
                    "vulkan adapter: op not yet implemented: {}",
                    op_name(other)
                )))
            }
        }
    }
    rec.finish().map_err(|e| be(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::graph::Graph;
    use infr_core::tensor::TensorDesc;
    use infr_core::DType;

    /// Prove the adapter machinery end-to-end (compile → bind → execute → download): a one-op
    /// `RmsNorm` graph run through the Vulkan seam must match a host reference. (Milestone #2: a
    /// small graph runs on Vulkan.)
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn rmsnorm_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (rows, dim, eps) = (2usize, 8usize, 1e-6f32);
        let x: Vec<f32> = (0..rows * dim).map(|i| i as f32 * 0.1 - 0.4).collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + i as f32 * 0.05).collect();
        // host reference rmsnorm
        let mut want = vec![0f32; rows * dim];
        for r in 0..rows {
            let b = r * dim;
            let ss = (0..dim).map(|i| x[b + i] * x[b + i]).sum::<f32>() / dim as f32;
            let s = 1.0 / (ss + eps).sqrt();
            for i in 0..dim {
                want[b + i] = x[b + i] * s * w[i];
            }
        }
        // build the graph
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![dim], DType::F32));
        let yi = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
        g.push(Op::RmsNorm {
            x: xi,
            weight: wi,
            dst: yi,
            rows: rows as u32,
            dim: dim as u32,
            eps,
        });
        // device buffers + bind
        let xb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let wb = be_.alloc(dim * 4, BufferUsage::Weights).unwrap();
        let yb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        be_.upload(wb.as_ref(), bytemuck::cast_slice(&w)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(wi, wb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; rows * dim];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..rows * dim {
            assert!(
                (got[i] - want[i]).abs() < 1e-3,
                "rmsnorm mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }
}

/// Short op name for the "not yet implemented" error (until every op is mapped).
fn op_name(op: &Op) -> &'static str {
    match op {
        Op::RmsNorm { .. } => "RmsNorm",
        Op::Linear { .. } => "Linear",
        Op::QkNorm { .. } => "QkNorm",
        Op::Rope { .. } => "Rope",
        Op::QkNormRope { .. } => "QkNormRope",
        Op::WriteKv { .. } => "WriteKv",
        Op::Attention { .. } => "Attention",
        Op::GatedAct { .. } => "GatedAct",
        Op::MoeFfn { .. } => "MoeFfn",
        Op::Conv1dSilu { .. } => "Conv1dSilu",
        Op::DeltaNet { .. } => "DeltaNet",
        Op::Add { .. } => "Add",
        Op::Scale { .. } => "Scale",
        Op::Softcap { .. } => "Softcap",
        Op::Copy { .. } => "Copy",
    }
}
