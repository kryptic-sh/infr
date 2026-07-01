//! Graph execution: host-side staging (identical to the CPU interpreter) with each op's arithmetic
//! dispatched to a Metal compute kernel. See the crate docs for why this is host-orchestrated.

use crate::{metal_buf, MetalBackend};
use infr_core::backend::{Bindings, Plan};
use infr_core::error::Error;
use infr_core::graph::{Op, TensorKind};
use infr_core::tensor::{DType, TensorId};
use infr_core::Result;
use infr_gguf::dequant::dequant_block;
use metal::{Buffer as MtlBuffer, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use std::sync::Arc;

/// Reinterpret raw buffer bytes as `f32` per `dtype` (dequantizing quant/f16/bf16, widening integer
/// tensors). The host-side "read a tensor's value", matching `infr-cpu`'s `bytes_to_f32`.
fn bytes_to_f32(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        DType::I32 => bytemuck::cast_slice::<u8, i32>(bytes)
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DType::U32 => bytemuck::cast_slice::<u8, u32>(bytes)
            .iter()
            .map(|&v| v as f32)
            .collect(),
        other => dequant_block(other, bytes).expect("metal backend: host dequant"),
    }
}

/// Gated-FFN activation applied to the gate value (matches `infr-cpu`'s `act_fn`).
fn act_fn(act: infr_core::graph::Activation, g: f32) -> f32 {
    use infr_core::graph::Activation;
    match act {
        Activation::Silu => g / (1.0 + (-g).exp()),
        Activation::Gelu => 0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()),
        Activation::Sigmoid => 1.0 / (1.0 + (-g).exp()),
    }
}

impl MetalBackend {
    // ---- small device-buffer helpers (StorageModeShared = CPU-visible on Apple Silicon) ----

    /// A device buffer initialized from an f32 host slice.
    fn f32_buf(&self, data: &[f32]) -> MtlBuffer {
        let len = (data.len().max(1) * 4) as u64;
        let buf = self
            .device
            .new_buffer(len, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut f32, data.len());
        }
        buf
    }

    /// A zeroed device buffer of `n` f32s.
    fn zeros_buf(&self, n: usize) -> MtlBuffer {
        self.device
            .new_buffer((n.max(1) * 4) as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Read `n` f32s out of a device buffer.
    fn read_f32(buf: &MtlBuffer, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), n);
        }
        out
    }

    /// Read the first `n` f16 elements of a buffer, widening to f32.
    fn read_f16_prefix(buf: &crate::MetalBuffer, n: usize) -> Vec<f32> {
        let ptr = buf.raw.contents() as *const u16;
        (0..n)
            .map(|i| half::f16::from_bits(unsafe { *ptr.add(i) }).to_f32())
            .collect()
    }

    /// Encode one kernel dispatch and block until it completes (correctness-first: every op is its
    /// own command buffer, so ops run strictly in graph order with no manual barriers). `data_bufs`
    /// bind to buffer indices `0..k`; `params` (packed scalars) binds at index `k`.
    fn dispatch(
        &self,
        pso: &ComputePipelineState,
        data_bufs: &[&MtlBuffer],
        params: &[u8],
        threads: usize,
    ) {
        if threads == 0 {
            return;
        }
        objc::rc::autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pso);
            for (i, b) in data_bufs.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            if !params.is_empty() {
                enc.set_bytes(
                    data_bufs.len() as u64,
                    params.len() as u64,
                    params.as_ptr() as *const c_void,
                );
            }
            let tg = (pso.max_total_threads_per_threadgroup() as usize).min(threads.max(1)) as u64;
            enc.dispatch_threads(MTLSize::new(threads as u64, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });
    }

    pub(crate) fn execute_graph(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        let g = &plan
            .as_any()
            .downcast_ref::<infr_core::backend::GraphPlan>()
            .expect("metal backend: plan is not a GraphPlan")
            .graph;

        // f32 working store for every Input/Internal/Output handle (mirrors the CPU interpreter).
        // KV caches are written/read in place from their bound buffers (see `direct`).
        let direct = g.in_place_inputs();
        let mut vals: Vec<Vec<f32>> = vec![Vec::new(); g.tensors.len()];
        for (i, decl) in g.tensors.iter().enumerate() {
            match decl.kind {
                TensorKind::Internal | TensorKind::Output => {
                    vals[i] = vec![0f32; decl.desc.numel()]
                }
                TensorKind::Input if direct.contains(&TensorId(i as u32)) => {}
                TensorKind::Input => {
                    let buf = metal_buf(
                        bindings
                            .get(TensorId(i as u32))
                            .expect("metal backend: unbound Input"),
                    );
                    let bytes = Self::read_bytes(buf);
                    vals[i] = bytes_to_f32(&bytes, decl.desc.dtype);
                }
                TensorKind::Weight => {}
            }
        }

        for op in &g.ops {
            self.run_op(op, g, bindings, &mut vals)?;
        }

        // Write back Outputs (and mutated f32 Inputs, e.g. recurrent state) to their bound buffers.
        for (i, decl) in g.tensors.iter().enumerate() {
            let write_back = matches!(decl.kind, TensorKind::Output)
                || (decl.kind == TensorKind::Input
                    && decl.desc.dtype == DType::F32
                    && !direct.contains(&TensorId(i as u32)));
            if !write_back {
                continue;
            }
            if let Some(buf) = bindings.get(TensorId(i as u32)) {
                let b = metal_buf(buf);
                let src: &[u8] = bytemuck::cast_slice(&vals[i]);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        b.raw.contents() as *mut u8,
                        src.len().min(b.len),
                    );
                }
            }
        }
        Ok(())
    }

    /// Fetch a dequantized weight as a device f32 buffer, cached by bound-buffer address.
    fn weight_buf(
        &self,
        id: TensorId,
        g: &infr_core::graph::Graph,
        bindings: &Bindings,
    ) -> Arc<MtlBuffer> {
        let buf = metal_buf(bindings.get(id).expect("metal backend: unbound Weight"));
        let key = buf as *const _ as usize;
        if let Some(w) = self.weight_cache.lock().unwrap().get(&key) {
            return w.clone();
        }
        let bytes = Self::read_bytes(buf);
        let f = bytes_to_f32(&bytes, g.desc(id).dtype);
        let w = Arc::new(self.f32_buf(&f));
        self.weight_cache.lock().unwrap().insert(key, w.clone());
        w
    }

    fn read_bytes(buf: &crate::MetalBuffer) -> Vec<u8> {
        let mut v = vec![0u8; buf.len];
        unsafe {
            std::ptr::copy_nonoverlapping(buf.raw.contents() as *const u8, v.as_mut_ptr(), v.len());
        }
        v
    }

    /// Read `len` bytes at `off` from a buffer's (unified-memory) contents.
    fn read_bytes_range(buf: &crate::MetalBuffer, off: usize, len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (buf.raw.contents() as *const u8).add(off),
                v.as_mut_ptr(),
                len,
            );
        }
        v
    }

    /// Dequantize a bound weight to host f32 (small weights: norm/conv/router-adjacent gains).
    fn weight_host(
        &self,
        id: TensorId,
        g: &infr_core::graph::Graph,
        bindings: &Bindings,
    ) -> Vec<f32> {
        let buf = metal_buf(bindings.get(id).expect("metal backend: unbound Weight"));
        bytes_to_f32(&Self::read_bytes(buf), g.desc(id).dtype)
    }

    /// Single-row matmul on the GPU: `out[out_f] = x[in_f] · Wᵀ`, W row-major [out_f, in_f] f32.
    fn gpu_matvec(&self, x: &[f32], w: &[f32], in_f: usize, out_f: usize) -> Result<Vec<f32>> {
        let bx = self.f32_buf(x);
        let bw = self.f32_buf(w);
        let bd = self.zeros_buf(out_f);
        let pso = self.pipelines.get("linear_f32")?;
        let mut p = 1u32.to_ne_bytes().to_vec(); // m = 1
        p.extend_from_slice(&(in_f as u32).to_ne_bytes());
        p.extend_from_slice(&(out_f as u32).to_ne_bytes());
        self.dispatch(&pso, &[&bx, &bw, &bd], &p, out_f);
        Ok(Self::read_f32(&bd, out_f))
    }

    fn run_op(
        &self,
        op: &Op,
        g: &infr_core::graph::Graph,
        bindings: &Bindings,
        vals: &mut [Vec<f32>],
    ) -> Result<()> {
        match *op {
            Op::RmsNorm {
                x,
                weight,
                dst,
                rows,
                dim,
                eps,
            } => {
                let (rows, dim) = (rows as usize, dim as usize);
                let bx = self.f32_buf(&vals[x.0 as usize]);
                let bw = self.weight_buf(weight, g, bindings);
                let bd = self.zeros_buf(rows * dim);
                let pso = self.pipelines.get("rmsnorm_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(dim as u32).to_ne_bytes());
                p.extend_from_slice(&eps.to_ne_bytes());
                self.dispatch(&pso, &[&bx, bw.as_ref(), &bd], &p, rows);
                vals[dst.0 as usize] = Self::read_f32(&bd, rows * dim);
            }
            Op::QkNorm {
                x,
                weight,
                dst,
                rows,
                n_head,
                head_dim,
                eps,
            } => {
                let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                let bx = self.f32_buf(&vals[x.0 as usize]);
                let bw = self.weight_buf(weight, g, bindings);
                let bd = self.zeros_buf(rows * nh * hd);
                let pso = self.pipelines.get("qknorm_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&eps.to_ne_bytes());
                self.dispatch(&pso, &[&bx, bw.as_ref(), &bd], &p, rows * nh);
                vals[dst.0 as usize] = Self::read_f32(&bd, rows * nh * hd);
            }
            Op::Linear {
                x,
                weight,
                dst,
                m,
                in_f,
                out_f,
            } => {
                let (m, in_f, out_f) = (m as usize, in_f as usize, out_f as usize);
                let bx = self.f32_buf(&vals[x.0 as usize]);
                let bw = self.weight_buf(weight, g, bindings); // dequantized f32, cached
                let bd = self.zeros_buf(m * out_f);
                let pso = self.pipelines.get("linear_f32")?;
                let mut p = (m as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(in_f as u32).to_ne_bytes());
                p.extend_from_slice(&(out_f as u32).to_ne_bytes());
                self.dispatch(&pso, &[&bx, bw.as_ref(), &bd], &p, m * out_f);
                vals[dst.0 as usize] = Self::read_f32(&bd, m * out_f);
            }
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
                let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                let bx = self.f32_buf(&vals[x.0 as usize]);
                let bpos = self.f32_buf(&vals[positions.0 as usize]);
                let bff = match freq_factors {
                    Some(f) => self.f32_buf(&vals[f.0 as usize]),
                    None => self.zeros_buf(1),
                };
                let bd = self.zeros_buf(rows * nh * hd);
                let pso = self.pipelines.get("rope_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&(rope_dim).to_ne_bytes());
                p.extend_from_slice(&theta.to_ne_bytes());
                p.extend_from_slice(&(freq_factors.is_some() as u32).to_ne_bytes());
                self.dispatch(&pso, &[&bx, &bpos, &bff, &bd], &p, rows * nh);
                vals[dst.0 as usize] = Self::read_f32(&bd, rows * nh * hd);
            }
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
                let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                let bx = self.f32_buf(&vals[x.0 as usize]);
                let bw = self.weight_buf(weight, g, bindings);
                let bpos = self.f32_buf(&vals[positions.0 as usize]);
                let bff = match freq_factors {
                    Some(f) => self.f32_buf(&vals[f.0 as usize]),
                    None => self.zeros_buf(1),
                };
                let bd = self.zeros_buf(rows * nh * hd);
                let pso = self.pipelines.get("qknormrope_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&(rope_dim).to_ne_bytes());
                p.extend_from_slice(&theta.to_ne_bytes());
                p.extend_from_slice(&eps.to_ne_bytes());
                p.extend_from_slice(&(freq_factors.is_some() as u32).to_ne_bytes());
                self.dispatch(&pso, &[&bx, bw.as_ref(), &bpos, &bff, &bd], &p, rows * nh);
                vals[dst.0 as usize] = Self::read_f32(&bd, rows * nh * hd);
            }
            Op::Add { a, b, dst, n } => {
                let n = n as usize;
                let ba = self.f32_buf(&vals[a.0 as usize]);
                let bb = self.f32_buf(&vals[b.0 as usize]);
                let bd = self.zeros_buf(n);
                let pso = self.pipelines.get("add_f32")?;
                self.dispatch(&pso, &[&ba, &bb, &bd], &(n as u32).to_ne_bytes(), n);
                vals[dst.0 as usize] = Self::read_f32(&bd, n);
            }
            Op::Scale { x, dst, s, n } => {
                let n = n as usize;
                let bx = self.f32_buf(&vals[x.0 as usize]);
                let bd = self.zeros_buf(n);
                let pso = self.pipelines.get("scale_f32")?;
                let mut p = s.to_ne_bytes().to_vec();
                p.extend_from_slice(&(n as u32).to_ne_bytes());
                self.dispatch(&pso, &[&bx, &bd], &p, n);
                vals[dst.0 as usize] = Self::read_f32(&bd, n);
            }
            Op::Softcap { x, dst, cap, n } => {
                let n = n as usize;
                let bx = self.f32_buf(&vals[x.0 as usize]);
                let bd = self.zeros_buf(n);
                let pso = self.pipelines.get("softcap_f32")?;
                let mut p = cap.to_ne_bytes().to_vec();
                p.extend_from_slice(&(n as u32).to_ne_bytes());
                self.dispatch(&pso, &[&bx, &bd], &p, n);
                vals[dst.0 as usize] = Self::read_f32(&bd, n);
            }
            Op::GatedAct {
                gate,
                up,
                dst,
                rows,
                nff,
                act,
                up_off,
            } => {
                let (rows, nff) = (rows as usize, nff as usize);
                let bg = self.f32_buf(&vals[gate.0 as usize]);
                let bu = self.f32_buf(&vals[up.0 as usize]);
                let bd = self.zeros_buf(rows * nff);
                let pso = self.pipelines.get("gatedact_f32")?;
                let act_code: u32 = match act {
                    infr_core::graph::Activation::Silu => 0,
                    infr_core::graph::Activation::Sigmoid => 2,
                    infr_core::graph::Activation::Gelu => 1,
                };
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nff as u32).to_ne_bytes());
                p.extend_from_slice(&act_code.to_ne_bytes());
                p.extend_from_slice(&up_off.to_ne_bytes());
                self.dispatch(&pso, &[&bg, &bu, &bd], &p, rows * nff);
                vals[dst.0 as usize] = Self::read_f32(&bd, rows * nff);
            }
            Op::WriteKv {
                src,
                cache,
                rows,
                row_stride,
                pos,
            } => {
                // Stateful write into the persistent (unified-memory) KV buffer — cast to the cache
                // dtype on write, touching only `rows` rows. Host-side, matching the CPU reference.
                let (rows, rs, pos) = (rows as usize, row_stride as usize, pos as usize);
                let s = &vals[src.0 as usize];
                let cbuf = metal_buf(
                    bindings
                        .get(cache)
                        .expect("metal backend: unbound KV cache"),
                );
                let base = pos * rs;
                let n = rows * rs;
                match g.desc(cache).dtype {
                    DType::F16 => {
                        let ptr = cbuf.raw.contents() as *mut u16;
                        for (i, &v) in s[..n].iter().enumerate() {
                            unsafe {
                                *ptr.add(base + i) = half::f16::from_f32(v).to_bits();
                            }
                        }
                    }
                    _ => {
                        let ptr = cbuf.raw.contents() as *mut f32;
                        unsafe {
                            std::ptr::copy_nonoverlapping(s.as_ptr(), ptr.add(base), n);
                        }
                    }
                }
            }
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
                let (rows, kv_len, nh, nkv, hd) = (
                    rows as usize,
                    kv_len as usize,
                    n_head as usize,
                    n_kv as usize,
                    head_dim as usize,
                );
                if hd > 256 {
                    return Err(Error::Unsupported(format!(
                        "metal attention: head_dim {hd} exceeds MAX_HD 256 (shader acc[] cap)"
                    )));
                }
                // Materialize the valid KV prefix (kv_len rows) as f32, dequantizing an f16 cache —
                // the inner dot runs in f32 either way. Read straight from the bound cache buffers.
                let need = kv_len * nkv * hd;
                let kbuf = metal_buf(bindings.get(k_cache).expect("metal: unbound k_cache"));
                let vbuf = metal_buf(bindings.get(v_cache).expect("metal: unbound v_cache"));
                let (ks, vs) = match g.desc(k_cache).dtype {
                    DType::F16 => (
                        Self::read_f16_prefix(kbuf, need),
                        Self::read_f16_prefix(vbuf, need),
                    ),
                    _ => (
                        Self::read_f32(&kbuf.raw, need),
                        Self::read_f32(&vbuf.raw, need),
                    ),
                };
                let bq = self.f32_buf(&vals[q.0 as usize]);
                let bk = self.f32_buf(&ks);
                let bv = self.f32_buf(&vs);
                let bd = self.zeros_buf(rows * nh * hd);
                let window: u32 = match mask {
                    infr_core::graph::AttnMask::Causal => 0,
                    infr_core::graph::AttnMask::SlidingWindow(w) => w as u32,
                };
                let pso = self.pipelines.get("attention_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(kv_len as u32).to_ne_bytes());
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(nkv as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&scale.to_ne_bytes());
                p.extend_from_slice(&window.to_ne_bytes());
                p.extend_from_slice(&pos.to_ne_bytes());
                self.dispatch(&pso, &[&bq, &bk, &bv, &bd], &p, rows * nh);
                vals[dst.0 as usize] = Self::read_f32(&bd, rows * nh * hd);
            }
            Op::MoeFfn {
                x,
                router,
                gate_exps,
                up_exps,
                down_exps,
                dst,
                ne,
                n_expert,
                n_used,
                n_ff_exp,
                scale,
                act,
            } => {
                let (ne, n_expert, n_used, nffx) = (
                    ne as usize,
                    n_expert as usize,
                    n_used as usize,
                    n_ff_exp as usize,
                );
                let xs = vals[x.0 as usize].clone();
                // Router (host, mirroring the CPU reference). Structurally the same top-k selection,
                // but the logits use a naive f32 sum here vs the CPU's 8-accumulator dot, so the
                // summation order differs — top-k can pick differently on a near-tie logit.
                let rw = self.weight_host(router, g, bindings);
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| (0..ne).map(|i| rw[e * ne + i] * xs[i]).sum::<f32>())
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                for p in probs.iter_mut() {
                    *p /= psum;
                }
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| {
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                // Per-expert stacked-weight byte-slice sizes (each expert = an equal slice).
                let gbuf = metal_buf(bindings.get(gate_exps).expect("metal: unbound gate_exps"));
                let ubuf = metal_buf(bindings.get(up_exps).expect("metal: unbound up_exps"));
                let dbuf = metal_buf(bindings.get(down_exps).expect("metal: unbound down_exps"));
                let (gst, ust, dsz) = (
                    gbuf.len / n_expert,
                    ubuf.len / n_expert,
                    dbuf.len / n_expert,
                );
                let (gdt, udt, ddt) = (
                    g.desc(gate_exps).dtype,
                    g.desc(up_exps).dtype,
                    g.desc(down_exps).dtype,
                );
                let mut out = vec![0f32; ne];
                for &e in &idx {
                    // Dequant only this expert's slices, then matvec on the GPU.
                    let gw = bytes_to_f32(&Self::read_bytes_range(gbuf, e * gst, gst), gdt);
                    let uw = bytes_to_f32(&Self::read_bytes_range(ubuf, e * ust, ust), udt);
                    let dw = bytes_to_f32(&Self::read_bytes_range(dbuf, e * dsz, dsz), ddt);
                    let gate = self.gpu_matvec(&xs, &gw, ne, nffx)?;
                    let up = self.gpu_matvec(&xs, &uw, ne, nffx)?;
                    let actv: Vec<f32> = (0..nffx).map(|i| act_fn(act, gate[i]) * up[i]).collect();
                    let y = self.gpu_matvec(&actv, &dw, nffx, ne)?;
                    let w_e = probs[e] / wsum * scale;
                    for i in 0..ne {
                        out[i] += w_e * y[i];
                    }
                }
                vals[dst.0 as usize] = out;
            }
            // Sequential per-token recurrences (control-flow heavy, tiny) — host-side, exactly
            // mirroring the CPU reference. The recurrent `state` is a bound f32 Input read/written
            // through `vals` (loaded in the preamble, written back after the op loop).
            Op::Conv1dSilu {
                x,
                weight,
                state,
                dst,
                channels,
                kernel,
            } => {
                let (cc, kk) = (channels as usize, kernel as usize);
                let xs = vals[x.0 as usize].clone();
                let ws = self.weight_host(weight, g, bindings); // [channels, kernel]
                let st = &mut vals[state.0 as usize]; // [(kernel-1), channels], oldest first
                let mut out = vec![0f32; cc];
                for ch in 0..cc {
                    let mut acc = 0f32;
                    for j in 0..kk - 1 {
                        acc += st[j * cc + ch] * ws[ch * kk + j];
                    }
                    acc += xs[ch] * ws[ch * kk + (kk - 1)];
                    out[ch] = acc / (1.0 + (-acc).exp()); // silu
                }
                for j in 0..kk.saturating_sub(2) {
                    for ch in 0..cc {
                        st[j * cc + ch] = st[(j + 1) * cc + ch];
                    }
                }
                if kk >= 2 {
                    for ch in 0..cc {
                        st[(kk - 2) * cc + ch] = xs[ch];
                    }
                }
                vals[dst.0 as usize] = out;
            }
            Op::DeltaNet {
                q,
                k,
                v,
                b,
                a,
                a_coef,
                dt_bias,
                state,
                dst,
                n_vhead,
                n_khead,
                head_k,
                head_v,
                eps,
            } => {
                let (nv, nk, kd, vd) = (
                    n_vhead as usize,
                    n_khead as usize,
                    head_k as usize,
                    head_v as usize,
                );
                let qf = vals[q.0 as usize].clone();
                let kf = vals[k.0 as usize].clone();
                let vf = vals[v.0 as usize].clone();
                let bf = vals[b.0 as usize].clone();
                let af = vals[a.0 as usize].clone();
                let acoef = self.weight_host(a_coef, g, bindings);
                let dtb = self.weight_host(dt_bias, g, bindings);
                let st = &mut vals[state.0 as usize]; // [nv, kd, vd]
                let mut out = vec![0f32; nv * vd];
                let qscale = 1.0 / (kd as f32).sqrt();
                let l2 = |slice: &[f32]| -> f32 {
                    (slice.iter().map(|x| x * x).sum::<f32>() + eps).sqrt()
                };
                for h in 0..nv {
                    let kh_idx = h % nk; // GQA: q/k heads tiled to nv value heads
                    let mut qh = qf[kh_idx * kd..kh_idx * kd + kd].to_vec();
                    let mut kh = kf[kh_idx * kd..kh_idx * kd + kd].to_vec();
                    let vh = &vf[h * vd..h * vd + vd];
                    let qn = l2(&qh);
                    let kn = l2(&kh);
                    for x in qh.iter_mut() {
                        *x = *x / qn * qscale;
                    }
                    for x in kh.iter_mut() {
                        *x /= kn;
                    }
                    let beta = 1.0 / (1.0 + (-bf[h]).exp());
                    let sp = {
                        let z = af[h] + dtb[h];
                        z.max(0.0) + (-z.abs()).exp().ln_1p()
                    };
                    let decay = (acoef[h] * sp).exp();
                    let sh = &mut st[h * kd * vd..(h + 1) * kd * vd]; // [kd, vd]
                    for x in sh.iter_mut() {
                        *x *= decay;
                    }
                    let mut kv = vec![0f32; vd];
                    for kk in 0..kd {
                        let kkv = kh[kk];
                        let row = &sh[kk * vd..kk * vd + vd];
                        for d in 0..vd {
                            kv[d] += kkv * row[d];
                        }
                    }
                    let delta: Vec<f32> = (0..vd).map(|d| (vh[d] - kv[d]) * beta).collect();
                    for kk in 0..kd {
                        let kkv = kh[kk];
                        let row = &mut sh[kk * vd..kk * vd + vd];
                        for d in 0..vd {
                            row[d] += kkv * delta[d];
                        }
                    }
                    let oh = &mut out[h * vd..h * vd + vd];
                    for kk in 0..kd {
                        let qv = qh[kk];
                        let row = &sh[kk * vd..kk * vd + vd];
                        for d in 0..vd {
                            oh[d] += qv * row[d];
                        }
                    }
                }
                vals[dst.0 as usize] = out;
            }
            // Pure data movement (no arithmetic): done host-side, identical to the CPU reference.
            Op::Copy {
                src,
                src_off,
                dst,
                dst_off,
                n,
            } => {
                let (so, dof, n) = (src_off as usize, dst_off as usize, n as usize);
                let s = vals[src.0 as usize].clone();
                vals[dst.0 as usize][dof..dof + n].copy_from_slice(&s[so..so + n]);
            }
        }
        Ok(())
    }
}
