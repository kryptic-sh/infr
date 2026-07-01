//! Vulkan adapter for the agnostic `infr_core::Backend` seam: lower a `Graph` of composite ops onto
//! the existing fused Recorder kernels, recorded into one command buffer. Mirrors the CPU
//! interpreter (`infr-llama::cpu_backend`) op-for-op, but executes on the GPU — so the SAME model
//! `Graph` runs on either backend. Built incrementally; ops not yet mapped return an error.

use crate::linear::native_dense_supported;
use crate::{be, VulkanBackend};
use infr_core::backend::{Bindings, Buffer, BufferUsage, GraphPlan, Plan};
use infr_core::error::Result;
use infr_core::graph::{Activation, AttnMask, Graph, Op, TensorKind};
use infr_core::{Backend, TensorId};

pub(crate) fn compile(graph: &Graph) -> Result<Box<dyn Plan>> {
    // The Graph is replayed each `execute` (buffers re-bound per step, no recompile) — the shared
    // GraphPlan carries it. A later pass can pre-record a resubmittable command buffer keyed by shape.
    Ok(GraphPlan::boxed(graph))
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
        .downcast_ref::<GraphPlan>()
        .ok_or_else(|| be("vulkan adapter: plan is not a GraphPlan"))?
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

    // RoPE position: the static `qk_norm_rope`/`rope` kernels take a scalar `rope_pos`, but the IR
    // carries a `positions` i32 tensor. Read `positions[0]` (decode rows=1, or the start of a
    // consecutive-prefill run) up front — `download` syncs, so it must precede the recorder.
    let mut rope_pos: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for op in &graph.ops {
        let pid = match op {
            Op::Rope { positions, .. } | Op::QkNormRope { positions, .. } => Some(*positions),
            _ => None,
        };
        if let Some(pid) = pid {
            if let std::collections::hash_map::Entry::Vacant(e) = rope_pos.entry(pid.0) {
                let mut b = [0u8; 4];
                be_.download(r(pid)?, &mut b)?;
                e.insert(i32::from_le_bytes(b) as usize);
            }
        }
    }

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
            // `dst = x · Wᵀ` — dispatch by weight dtype: native-quant blocks (in-shader dequant GEMV)
            // or f16. (Prefill GEMM / mmq tuning is a later perf pass; this is the decode GEMV.)
            Op::Linear {
                x,
                weight,
                dst,
                m,
                in_f,
                out_f,
            } => {
                let (m, in_f, out_f) = (*m as usize, *in_f as usize, *out_f as usize);
                let (w, xb, y) = (r(*weight)?, r(*x)?, r(*dst)?);
                let dt = graph.desc(*weight).dtype;
                if native_dense_supported(dt) {
                    rec.linear_native(dt, w, xb, y, m, in_f, out_f);
                } else {
                    rec.linear(w, xb, y, m, in_f, out_f);
                }
            }
            Op::Add { a, b, dst, n } => rec.add(r(*a)?, r(*b)?, r(*dst)?, *n as usize),
            Op::Scale { x, dst, s, n } => {
                let n = *n as usize;
                // recorder `scale` is in place on its buffer; copy x→dst first if they differ.
                if x != dst {
                    let eb = graph.desc(*dst).dtype.dense_bytes(1).unwrap_or(4);
                    rec.copy(r(*x)?, 0, r(*dst)?, 0, n * eb);
                }
                rec.scale(r(*dst)?, *s, n);
            }
            Op::Copy {
                src,
                src_off,
                dst,
                dst_off,
                n,
            } => {
                // IR offsets/counts are in ELEMENTS; the recorder copy takes BYTES.
                let eb = graph.desc(*src).dtype.dense_bytes(1).unwrap_or(4);
                rec.copy(
                    r(*src)?,
                    *src_off as usize * eb,
                    r(*dst)?,
                    *dst_off as usize * eb,
                    *n as usize * eb,
                );
            }
            // Gated FFN activation: `act(gate) * up[+up_off]`. up_off (E2B per-layer slice) only
            // arises with Gelu (gemma); silu/sigmoid are always up_off==0.
            Op::GatedAct {
                gate,
                up,
                dst,
                rows,
                nff,
                act,
                up_off,
            } => {
                let n = *rows as usize * *nff as usize;
                let (g_, u_, y) = (r(*gate)?, r(*up)?, r(*dst)?);
                match act {
                    Activation::Silu => {
                        if *up_off != 0 {
                            return Err(be("vulkan adapter: GatedAct Silu up_off!=0 unsupported"));
                        }
                        rec.silu_mul(g_, u_, y, n);
                    }
                    Activation::Sigmoid => {
                        if *up_off != 0 {
                            return Err(be(
                                "vulkan adapter: GatedAct Sigmoid up_off!=0 unsupported",
                            ));
                        }
                        rec.mul_sigmoid(g_, u_, y, n);
                    }
                    Activation::Gelu => {
                        let eb = graph.desc(*up).dtype.dense_bytes(1).unwrap_or(4);
                        rec.gelu_mul_off(g_, u_, *up_off as usize * eb, y, n);
                    }
                }
            }
            // Append a row into the persistent KV cache at row `pos`. store_f16 casts f32→f16 (the
            // common case: V / f32 K); an already-f16 source is a straight copy.
            Op::WriteKv {
                src,
                cache,
                rows,
                row_stride,
                pos,
            } => {
                let (rows, rs, pos) = (*rows as usize, *row_stride as usize, *pos as usize);
                let n = rows * rs;
                let (s, c) = (r(*src)?, r(*cache)?);
                match graph.desc(*src).dtype {
                    infr_core::DType::F16 => rec.copy(s, 0, c, pos * rs * 2, n * 2),
                    _ => rec.store_f16(s, c, n, pos * rs),
                }
            }
            // Fused per-head RMSNorm + RoPE → the GPU's fused kernel (f32 in → f16 out, so `dst` is an
            // f16 tensor). `freq_factors` = gemma4 proportional RoPE.
            Op::QkNormRope {
                x,
                weight,
                positions,
                dst,
                rows,
                n_head,
                head_dim,
                rope_dim,
                theta,
                eps,
                freq_factors,
            } => {
                let ff = match freq_factors {
                    Some(f) => Some(r(*f)?),
                    None => None,
                };
                rec.qk_norm_rope(
                    r(*x)?,
                    r(*weight)?,
                    r(*dst)?,
                    *rows as usize,
                    *n_head as usize,
                    *head_dim as usize,
                    *rope_dim as usize,
                    *theta,
                    rope_pos[&positions.0],
                    0,
                    *eps,
                    ff,
                );
            }
            // Standalone RoPE (llama: no q/k-norm). The basic kernel has no freq_factors — gemma4's
            // proportional RoPE always rides on QkNormRope, so a lone freq_factors RoPE is unexpected.
            Op::Rope {
                x,
                positions,
                dst,
                rows,
                n_head,
                head_dim,
                rope_dim,
                theta,
                freq_factors,
            } => {
                if freq_factors.is_some() {
                    return Err(be(
                        "vulkan adapter: standalone Rope with freq_factors unsupported",
                    ));
                }
                rec.rope(
                    r(*x)?,
                    r(*dst)?,
                    *rows as usize,
                    *n_head as usize,
                    *head_dim as usize,
                    *rope_dim as usize,
                    *theta,
                    rope_pos[&positions.0],
                );
            }
            // GQA scaled-dot-product attention over the f16 KV cache (causal / sliding-window).
            Op::Attention {
                q,
                k_cache,
                v_cache,
                dst,
                rows,
                kv_len,
                n_head,
                n_kv,
                head_dim,
                scale,
                mask,
                pos,
            } => {
                let window = match mask {
                    AttnMask::Causal => 0,
                    AttnMask::SlidingWindow(w) => *w,
                };
                rec.attention_kv(
                    r(*q)?,
                    r(*k_cache)?,
                    r(*v_cache)?,
                    r(*dst)?,
                    *rows as usize,
                    *kv_len as usize,
                    *n_head as usize,
                    *n_kv as usize,
                    *head_dim as usize,
                    *pos as usize,
                    window,
                    *scale,
                );
            }
            other => {
                return Err(be(format!(
                    "vulkan adapter: op not yet implemented: {}",
                    other.kind()
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

    /// A one-op `Linear` graph (f16 weight, 1-row GEMV) through the seam must match a host matvec.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn linear_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (in_f, out_f) = (16usize, 4usize);
        let x: Vec<f32> = (0..in_f).map(|i| i as f32 * 0.1 - 0.8).collect();
        let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32 * 0.03).sin()).collect();
        let wf16: Vec<u8> = w
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        // host reference uses the same f16-rounded weight the GPU reads
        let wq: Vec<f32> = wf16
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        let mut want = vec![0f32; out_f];
        for (o, wo) in want.iter_mut().enumerate() {
            *wo = (0..in_f).map(|i| x[i] * wq[o * in_f + i]).sum();
        }
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![1, in_f], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![out_f, in_f], DType::F16));
        let yi = g.output(TensorDesc::new(vec![1, out_f], DType::F32));
        g.push(Op::Linear {
            x: xi,
            weight: wi,
            dst: yi,
            m: 1,
            in_f: in_f as u32,
            out_f: out_f as u32,
        });
        let xb = be_.alloc(in_f * 4, BufferUsage::Activations).unwrap();
        let wb = be_.alloc(wf16.len(), BufferUsage::Weights).unwrap();
        let yb = be_.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        be_.upload(wb.as_ref(), &wf16).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(wi, wb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; out_f];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for o in 0..out_f {
            assert!(
                (got[o] - want[o]).abs() < 1e-2,
                "linear mismatch at {o}: got {} want {}",
                got[o],
                want[o]
            );
        }
    }

    /// A one-op `GatedAct` (SwiGLU: silu(gate)·up) graph through the seam must match a host loop.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn gated_act_silu_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let nff = 8usize;
        let gate: Vec<f32> = (0..nff).map(|i| i as f32 * 0.2 - 0.7).collect();
        let up: Vec<f32> = (0..nff).map(|i| 1.0 - i as f32 * 0.1).collect();
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let want: Vec<f32> = (0..nff).map(|i| silu(gate[i]) * up[i]).collect();
        let mut g = Graph::new();
        let gi = g.input(TensorDesc::new(vec![1, nff], DType::F32));
        let ui = g.input(TensorDesc::new(vec![1, nff], DType::F32));
        let yi = g.output(TensorDesc::new(vec![1, nff], DType::F32));
        g.push(Op::GatedAct {
            gate: gi,
            up: ui,
            dst: yi,
            rows: 1,
            nff: nff as u32,
            act: Activation::Silu,
            up_off: 0,
        });
        let gb = be_.alloc(nff * 4, BufferUsage::Activations).unwrap();
        let ub = be_.alloc(nff * 4, BufferUsage::Activations).unwrap();
        let yb = be_.alloc(nff * 4, BufferUsage::Activations).unwrap();
        be_.upload(gb.as_ref(), bytemuck::cast_slice(&gate))
            .unwrap();
        be_.upload(ub.as_ref(), bytemuck::cast_slice(&up)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(gi, gb.as_ref());
        bind.bind(ui, ub.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; nff];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..nff {
            assert!(
                (got[i] - want[i]).abs() < 1e-3,
                "gated_act mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// A one-op `QkNormRope` graph (per-head RMSNorm + RoPE, f32 in → f16 out, positions tensor) must
    /// match a host reference (f16 tolerance).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn qk_norm_rope_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (nh, hd, pos) = (2usize, 8usize, 3usize);
        let (eps, theta, rope_dim) = (1e-6f32, 10000.0f32, 8usize);
        let x: Vec<f32> = (0..nh * hd).map(|i| i as f32 * 0.1 - 0.5).collect();
        let w: Vec<f32> = (0..hd).map(|i| 1.0 + i as f32 * 0.05).collect();
        // host reference: per-head rmsnorm (×w) then split-half NEOX rope
        let mut want = vec![0f32; nh * hd];
        let hf = rope_dim / 2;
        for h in 0..nh {
            let b = h * hd;
            let ss = (0..hd).map(|i| x[b + i] * x[b + i]).sum::<f32>() / hd as f32;
            let s = 1.0 / (ss + eps).sqrt();
            let nrm: Vec<f32> = (0..hd).map(|i| x[b + i] * s * w[i]).collect();
            want[b..b + hd].copy_from_slice(&nrm);
            for p in 0..hf {
                let ang = pos as f32 * theta.powf(-2.0 * p as f32 / rope_dim as f32);
                let (sn, c) = (ang.sin(), ang.cos());
                want[b + p] = nrm[p] * c - nrm[p + hf] * sn;
                want[b + p + hf] = nrm[p] * sn + nrm[p + hf] * c;
            }
        }
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![1, nh, hd], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![hd], DType::F32));
        let pi = g.input(TensorDesc::new(vec![1], DType::I32));
        let yi = g.output(TensorDesc::new(vec![1, nh, hd], DType::F16));
        g.push(Op::QkNormRope {
            x: xi,
            weight: wi,
            positions: pi,
            dst: yi,
            rows: 1,
            n_head: nh as u32,
            head_dim: hd as u32,
            rope_dim: rope_dim as u32,
            theta,
            eps,
            freq_factors: None,
        });
        let xb = be_.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
        let wb = be_.alloc(hd * 4, BufferUsage::Weights).unwrap();
        let pb = be_.alloc(4, BufferUsage::Activations).unwrap();
        let yb = be_.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        be_.upload(wb.as_ref(), bytemuck::cast_slice(&w)).unwrap();
        be_.upload(pb.as_ref(), &(pos as i32).to_le_bytes())
            .unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(wi, wb.as_ref());
        bind.bind(pi, pb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut y16 = vec![0u8; nh * hd * 2];
        be_.download(yb.as_ref(), &mut y16).unwrap();
        let got: Vec<f32> = y16
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        for i in 0..nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 2e-2,
                "qk_norm_rope mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// A one-op `Attention` graph (GQA, causal, f16 q + f16 KV) must match a host softmax-attention.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn attention_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (nh, nkv, hd, pos) = (2usize, 1usize, 8usize, 2usize);
        let kv_len = pos + 1; // causal: query at `pos` attends keys 0..=pos
        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        let to_f16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect()
        };
        let deq = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        let q: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.07).sin()).collect();
        let k: Vec<f32> = (0..kv_len * nkv * hd)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..kv_len * nkv * hd)
            .map(|i| i as f32 * 0.01 - 0.1)
            .collect();
        let (qf, kf, vf) = (to_f16(&q), to_f16(&k), to_f16(&v));
        let (qd, kd, vd) = (deq(&qf), deq(&kf), deq(&vf)); // host uses the same f16-rounded values
                                                           // host GQA softmax attention
        let mut want = vec![0f32; nh * hd];
        for h in 0..nh {
            let kvh = h / group;
            let mut sc = vec![0f32; kv_len];
            let mut mx = f32::NEG_INFINITY;
            for (j, scj) in sc.iter_mut().enumerate() {
                let d: f32 = (0..hd)
                    .map(|x| qd[h * hd + x] * kd[(j * nkv + kvh) * hd + x])
                    .sum();
                *scj = d * scale;
                mx = mx.max(*scj);
            }
            let l: f32 = sc.iter().map(|s| (s - mx).exp()).sum();
            for (j, &s) in sc.iter().enumerate() {
                let p = (s - mx).exp() / l;
                for x in 0..hd {
                    want[h * hd + x] += p * vd[(j * nkv + kvh) * hd + x];
                }
            }
        }
        let mut g = Graph::new();
        let qi = g.input(TensorDesc::new(vec![1, nh, hd], DType::F16));
        let ki = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
        let vi = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
        let yi = g.output(TensorDesc::new(vec![1, nh, hd], DType::F32));
        g.push(Op::Attention {
            q: qi,
            k_cache: ki,
            v_cache: vi,
            dst: yi,
            rows: 1,
            kv_len: kv_len as u32,
            n_head: nh as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            scale,
            mask: AttnMask::Causal,
            pos: pos as u32,
        });
        let qb = be_.alloc(qf.len(), BufferUsage::Activations).unwrap();
        let kb = be_.alloc(kf.len(), BufferUsage::Activations).unwrap();
        let vb = be_.alloc(vf.len(), BufferUsage::Activations).unwrap();
        let yb = be_.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
        be_.upload(qb.as_ref(), &qf).unwrap();
        be_.upload(kb.as_ref(), &kf).unwrap();
        be_.upload(vb.as_ref(), &vf).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(qi, qb.as_ref());
        bind.bind(ki, kb.as_ref());
        bind.bind(vi, vb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; nh * hd];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 2e-2,
                "attention mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }
}
