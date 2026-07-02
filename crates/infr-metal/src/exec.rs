//! Graph execution: host-side staging (identical to the CPU interpreter) with each op's arithmetic
//! dispatched to a Metal compute kernel. See the crate docs for why this is host-orchestrated.

use crate::{metal_buf, MetalBackend};
use infr_core::backend::{Bindings, Plan};
use infr_core::error::Error;
use infr_core::graph::{Op, TensorKind};
use infr_core::tensor::{DType, TensorId};
use infr_core::Result;
use infr_gguf::dequant::dequant_block;
use metal::{
    Buffer as MtlBuffer, CommandBuffer, ComputeCommandEncoder, ComputePipelineState,
    MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::sync::Arc;

/// A quantized weight in the compact FACTORED device form (`infr_gguf::dequant::Factored`):
/// bit-packed codes (4/6/8-bit, chosen by the format's max code), one `(sc, m)` i16 pair per
/// 16-element block, and one `(d, dmin)` f16 pair per `dblk` elements, so
/// `weight = (d*sc)*code + (dmin*m)` — bit-for-bit the `dequant` reference (same f32 multiplies).
/// Q4_K lands at ~6.1 bpw resident and Q6_K at ~8.1, vs 32 for a dequant-to-f32 weight; decode
/// GEMV is bound on exactly this stream.
pub(crate) struct QuiWeight {
    codes: MtlBuffer,
    scm: MtlBuffer,
    dd: MtlBuffer,
    /// Kernel matching the code packing: `linear_quik4`/`linear_quik6`/`linear_quik8`.
    kern: &'static str,
    /// log2(elements per `(d, dmin)` pair): 5 for legacy 32-element blocks, 8 for K-quants.
    dshift: u32,
}

/// Where a tensor's current value lives. GPU ops keep their results on the device and only pay a
/// CPU↔GPU round-trip when a host-side op (or the final write-back) actually needs the bytes.
#[derive(Clone, Copy, PartialEq)]
enum Loc {
    Host,
    Device,
}

/// Per-forward execution state: the host mirror (`vals`), a device buffer per tensor (`dev`), where
/// each tensor currently lives (`loc`), and the open command buffer that batches consecutive GPU ops
/// so they share a single commit + wait instead of one per op.
struct Resident {
    vals: Vec<Vec<f32>>,
    dev: Vec<Option<Arc<MtlBuffer>>>,
    loc: Vec<Loc>,
    cb: Option<CommandBuffer>,
    enc: Option<ComputeCommandEncoder>,
}

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

/// Stable op label for the profiler.
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
        Op::Add { .. } => "Add",
        Op::Scale { .. } => "Scale",
        Op::Softcap { .. } => "Softcap",
        Op::Copy { .. } => "Copy",
        Op::CopyStrided { .. } => "CopyStrided",
        Op::MoeFfn { .. } => "MoeFfn",
        Op::Conv1dSilu { .. } => "Conv1dSilu",
        Op::DeltaNet { .. } => "DeltaNet",
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
            let t0 = self.profiling.then(std::time::Instant::now);
            cb.commit();
            cb.wait_until_completed();
            if let Some(t0) = t0 {
                // Wall time of one op's commit + GPU schedule + wait barrier.
                self.prof.lock().unwrap().add_dispatch(t0.elapsed());
            }
        });
    }

    /// Encode one kernel into the batch's shared command buffer (lazily opened) WITHOUT committing.
    /// Metal's automatic hazard tracking inserts the needed barriers between kernels that touch the
    /// same buffers, so the batch runs in graph order inside a single CPU↔GPU round-trip.
    fn encode(
        &self,
        r: &mut Resident,
        pso: &ComputePipelineState,
        bufs: &[&MtlBuffer],
        params: &[u8],
        threads: usize,
    ) {
        // Auto threadgroup: as wide as the pipeline allows (good for the big elementwise grids).
        self.encode_tg(r, pso, bufs, params, threads, 0);
    }

    /// As `encode`, but with an explicit threadgroup width (`tg`; 0 = auto). The simdgroup-GEMV
    /// kernels pass 32 (one simdgroup per threadgroup) so a matvec launches `out_f` threadgroups
    /// instead of a handful of wide ones — far more threadgroups for the GPU's cores to interleave,
    /// which hides the memory latency the tiny per-output dot would otherwise stall on.
    fn encode_tg(
        &self,
        r: &mut Resident,
        pso: &ComputePipelineState,
        bufs: &[&MtlBuffer],
        params: &[u8],
        threads: usize,
        tg: usize,
    ) {
        if threads == 0 {
            return;
        }
        if r.enc.is_none() {
            let cb = self.queue.new_command_buffer().to_owned();
            let enc = cb.new_compute_command_encoder().to_owned();
            r.cb = Some(cb);
            r.enc = Some(enc);
        }
        let enc = r.enc.as_ref().unwrap();
        enc.set_compute_pipeline_state(pso);
        for (i, b) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        if !params.is_empty() {
            enc.set_bytes(
                bufs.len() as u64,
                params.len() as u64,
                params.as_ptr() as *const c_void,
            );
        }
        let cap = pso.max_total_threads_per_threadgroup() as usize;
        let tg = if tg == 0 { cap } else { tg.min(cap) }.min(threads.max(1)) as u64;
        enc.dispatch_threads(MTLSize::new(threads as u64, 1, 1), MTLSize::new(tg, 1, 1));
    }

    /// Close the open batch: end encoding, commit, and wait. A no-op if nothing is buffered.
    fn flush(&self, r: &mut Resident) {
        if let Some(enc) = r.enc.take() {
            enc.end_encoding();
            let cb = r.cb.take().expect("metal: enc without command buffer");
            let t0 = self.profiling.then(std::time::Instant::now);
            cb.commit();
            cb.wait_until_completed();
            if let Some(t0) = t0 {
                self.prof.lock().unwrap().add_dispatch(t0.elapsed());
            }
        }
    }

    /// Ensure a tensor's value is a device buffer (uploading from the host mirror if needed), and
    /// return it. Tensors already produced on-device are returned as-is — no round-trip.
    fn ensure_device(&self, r: &mut Resident, id: TensorId) -> Arc<MtlBuffer> {
        let i = id.0 as usize;
        if r.dev[i].is_none() || r.loc[i] == Loc::Host {
            r.dev[i] = Some(Arc::new(self.f32_buf(&r.vals[i])));
            r.loc[i] = Loc::Device;
        }
        r.dev[i].clone().unwrap()
    }

    /// Get (or allocate) the persistent device output buffer for a tensor, sized for `n` f32s.
    fn dev_dst(&self, r: &mut Resident, id: TensorId, n: usize) -> Arc<MtlBuffer> {
        let i = id.0 as usize;
        let big_enough = matches!(&r.dev[i], Some(b) if b.length() as usize >= n * 4);
        if !big_enough {
            r.dev[i] = Some(Arc::new(self.zeros_buf(n)));
        }
        r.dev[i].clone().unwrap()
    }

    /// Ensure a tensor's value is in the host mirror `vals`, flushing the batch and downloading from
    /// the device if the latest value lives there. This is the only place a GPU→host stall happens.
    fn ensure_host(&self, r: &mut Resident, g: &infr_core::graph::Graph, id: TensorId) {
        let i = id.0 as usize;
        if r.loc[i] == Loc::Device {
            self.flush(r);
            let n = g.desc(id).numel();
            r.vals[i] = Self::read_f32(r.dev[i].as_ref().unwrap(), n);
            r.loc[i] = Loc::Host;
        }
    }

    pub(crate) fn execute_graph(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        let g = &plan
            .as_any()
            .downcast_ref::<infr_core::backend::GraphPlan>()
            .expect("metal backend: plan is not a GraphPlan")
            .graph;

        // Wrap the whole forward in one autorelease pool: the batched command buffers/encoders are
        // retained owned handles, so we drain the pool once per forward instead of once per op.
        objc::rc::autoreleasepool(|| self.run_graph(g, bindings))
    }

    fn run_graph(&self, g: &infr_core::graph::Graph, bindings: &Bindings) -> Result<()> {
        // f32 host mirror for every Input/Internal/Output handle (mirrors the CPU interpreter); GPU
        // results stay on-device until a host op or the write-back needs them (see `Loc`/`Resident`).
        // KV caches are written/read in place from their bound buffers (see `direct`).
        let direct = g.in_place_inputs();
        let n = g.tensors.len();
        let mut r = Resident {
            vals: vec![Vec::new(); n],
            dev: vec![None; n],
            loc: vec![Loc::Host; n],
            cb: None,
            enc: None,
        };
        for (i, decl) in g.tensors.iter().enumerate() {
            match decl.kind {
                TensorKind::Internal | TensorKind::Output => {
                    r.vals[i] = vec![0f32; decl.desc.numel()]
                }
                TensorKind::Input if direct.contains(&TensorId(i as u32)) => {}
                TensorKind::Input => {
                    let buf = metal_buf(
                        bindings
                            .get(TensorId(i as u32))
                            .expect("metal backend: unbound Input"),
                    );
                    let bytes = Self::read_bytes(buf);
                    r.vals[i] = bytes_to_f32(&bytes, decl.desc.dtype);
                }
                TensorKind::Weight => {}
            }
        }

        if self.profiling {
            for op in &g.ops {
                let t0 = std::time::Instant::now();
                self.run_op(op, g, bindings, &mut r)?;
                let enc = t0.elapsed();
                // Per-op mode: flush now so this op's GPU wall is isolable (breaks batching).
                let gpu = if self.prof_ops {
                    let tg = std::time::Instant::now();
                    self.flush(&mut r);
                    Some(tg.elapsed())
                } else {
                    None
                };
                let mut pr = self.prof.lock().unwrap();
                pr.add_op(op_name(op), enc);
                if let Some(gpu) = gpu {
                    pr.add_op_gpu(op_name(op), gpu);
                }
            }
            self.prof.lock().unwrap().add_forward();
        } else {
            for op in &g.ops {
                self.run_op(op, g, bindings, &mut r)?;
            }
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
            if bindings.get(TensorId(i as u32)).is_some() {
                // Pull the value back to the host mirror (flushes the batch if it's still on-device).
                self.ensure_host(&mut r, g, TensorId(i as u32));
                let b = metal_buf(bindings.get(TensorId(i as u32)).unwrap());
                let src: &[u8] = bytemuck::cast_slice(&r.vals[i]);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        b.raw.contents() as *mut u8,
                        src.len().min(b.len),
                    );
                }
            }
        }
        // Close any trailing batch (an Internal-only tail never pulled to host).
        self.flush(&mut r);
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

    /// A device buffer initialized from a raw byte slice.
    fn bytes_buf(&self, data: &[u8]) -> MtlBuffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            data.len().max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Build (or fetch from cache) a quantized weight in on-device form. Q4_K and Q6_K — the
    /// formats real checkpoints ship — have NATIVE kernels that decode the raw GGUF block bytes,
    /// so the bound weight buffer is used as-is: no host repack, no extra residency, and the
    /// weight stream stays at the format's true size (~4.5 / ~6.6 bpw vs the factored form's
    /// ~6.1 / ~8.1). Everything else goes through the factored form (`dequant_factored`), codes
    /// bit-packed to the narrowest width the format's max code fits. Decode GEMV is bound on the
    /// weight byte stream, so every bit shaved here is throughput.
    fn weight_qui(
        &self,
        id: TensorId,
        g: &infr_core::graph::Graph,
        bindings: &Bindings,
    ) -> Arc<QuiWeight> {
        let buf = metal_buf(bindings.get(id).expect("metal backend: unbound Weight"));
        let key = buf as *const _ as usize;
        if let Some(w) = self.qui_cache.lock().unwrap().get(&key) {
            return w.clone();
        }
        let native_kern = match g.desc(id).dtype {
            DType::Q4K => Some("linear_q4k"),
            DType::Q6K => Some("linear_q6k"),
            _ => None,
        };
        if let Some(kern) = native_kern {
            // The kernel never reads scm/dd for native formats; bind tiny dummies.
            let w = Arc::new(QuiWeight {
                codes: buf.raw.clone(),
                scm: self.zeros_buf(1),
                dd: self.zeros_buf(1),
                kern,
                dshift: 0,
            });
            self.qui_cache.lock().unwrap().insert(key, w.clone());
            return w;
        }
        let bytes = Self::read_bytes(buf);
        let f = infr_gguf::dequant::dequant_factored(g.desc(id).dtype, &bytes);
        let maxcode = f.codes.iter().copied().max().unwrap_or(0);
        let (kern, codes_bytes): (&'static str, Vec<u8>) = if maxcode < 16 {
            // two codes per byte, low nibble first
            (
                "linear_quik4",
                f.codes
                    .chunks_exact(2)
                    .map(|p| p[0] | (p[1] << 4))
                    .collect(),
            )
        } else if maxcode < 64 {
            // four codes per three bytes, LSB-first 6-bit stream
            (
                "linear_quik6",
                f.codes
                    .chunks_exact(4)
                    .flat_map(|c| {
                        [
                            c[0] | (c[1] << 6),
                            (c[1] >> 2) | (c[2] << 4),
                            (c[2] >> 4) | (c[3] << 2),
                        ]
                    })
                    .collect(),
            )
        } else {
            ("linear_quik8", f.codes)
        };
        let w = Arc::new(QuiWeight {
            codes: self.bytes_buf(&codes_bytes),
            scm: self.bytes_buf(bytemuck::cast_slice(&f.scm)),
            dd: self.bytes_buf(bytemuck::cast_slice(
                &f.dd.iter().map(|h| h.to_bits()).collect::<Vec<u16>>(),
            )),
            kern,
            dshift: f.dblk.trailing_zeros(),
        });
        self.qui_cache.lock().unwrap().insert(key, w.clone());
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
        self.dispatch(&pso, &[&bx, &bw, &bd], &p, out_f * 32); // 32 lanes/output — see linear_f32
        Ok(Self::read_f32(&bd, out_f))
    }

    fn run_op(
        &self,
        op: &Op,
        g: &infr_core::graph::Graph,
        bindings: &Bindings,
        r: &mut Resident,
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
                let bx = self.ensure_device(r, x);
                let bw = self.weight_buf(weight, g, bindings);
                let bd = self.dev_dst(r, dst, rows * dim);
                let pso = self.pipelines.get("rmsnorm_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(dim as u32).to_ne_bytes());
                p.extend_from_slice(&eps.to_ne_bytes());
                // One simdgroup (32 lanes) per row — see `rmsnorm_f32`.
                self.encode_tg(
                    r,
                    &pso,
                    &[bx.as_ref(), bw.as_ref(), bd.as_ref()],
                    &p,
                    rows * 32,
                    32,
                );
                r.loc[dst.0 as usize] = Loc::Device;
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
                let bx = self.ensure_device(r, x);
                let bw = self.weight_buf(weight, g, bindings);
                let bd = self.dev_dst(r, dst, rows * nh * hd);
                let pso = self.pipelines.get("qknorm_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&eps.to_ne_bytes());
                // One simdgroup per (row, head) — see `qknorm_f32`.
                self.encode_tg(
                    r,
                    &pso,
                    &[bx.as_ref(), bw.as_ref(), bd.as_ref()],
                    &p,
                    rows * nh * 32,
                    32,
                );
                r.loc[dst.0 as usize] = Loc::Device;
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
                let bx = self.ensure_device(r, x);
                let bd = self.dev_dst(r, dst, m * out_f);
                let mut p = (m as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(in_f as u32).to_ne_bytes());
                p.extend_from_slice(&(out_f as u32).to_ne_bytes());
                // 32 lanes (one simdgroup) per output element — see `linear_f32`/`linear_quik*`.
                if infr_gguf::dequant::is_quant(g.desc(weight).dtype) {
                    // Native quant: decode the compact factored weight inline — no f32 blow-up.
                    // Kernel matches the weight's code packing (4/6/8-bit). Multi-row (prefill)
                    // takes the row-tiled variant: one simdgroup per (8-row tile, output), each
                    // weight block decoded once for all 8 rows, instead of the GEMV kernel
                    // re-streaming the whole weight matrix once per row.
                    let qw = self.weight_qui(weight, g, bindings);
                    // sgs = simdgroups to launch; tg = threadgroup width. GEMM tiles 8x16 per
                    // simdgroup (4 simdgroups per threadgroup for the staging tile). The GEMM
                    // kernel requires its full 128-thread threadgroup (per-simdgroup staging
                    // tiles indexed by simdgroup_index_in_threadgroup), so it is gated on the
                    // pipeline's own thread cap — see the Attention arm for why that cap can
                    // drop below the kernel's need — and degrades to the row-tiled kernel.
                    let hmm_kern = match qw.kern {
                        "linear_quik4" => "linear_quik4_hmm",
                        "linear_quik6" => "linear_quik6_hmm",
                        "linear_q4k" => "linear_q4k_hmm",
                        "linear_q6k" => "linear_q6k_hmm",
                        _ => "linear_quik8_hmm",
                    };
                    let hmm_ok = m >= 16
                        && out_f % 16 == 0
                        && self
                            .pipelines
                            .get(hmm_kern)?
                            .max_total_threads_per_threadgroup()
                            >= 128;
                    p.extend_from_slice(&qw.dshift.to_ne_bytes());
                    if hmm_ok {
                        // Prefill GEMM runs on half fragments (see `HGEMM_KERNEL`): cast x to a
                        // transient f16 buffer first, then one GEMM dispatch reads it.
                        let xh = Arc::new(self.device.new_buffer(
                            (m * in_f * 2).max(4) as u64,
                            MTLResourceOptions::StorageModeShared,
                        ));
                        let cast = self.pipelines.get("cast_f32_f16")?;
                        let n = (m * in_f) as u32;
                        self.encode(r, &cast, &[bx.as_ref(), &xh], &n.to_ne_bytes(), m * in_f);
                        let pso = self.pipelines.get(hmm_kern)?;
                        self.encode_tg(
                            r,
                            &pso,
                            &[xh.as_ref(), &qw.codes, &qw.scm, &qw.dd, bd.as_ref()],
                            &p,
                            m.div_ceil(32) * (out_f / 16) * 32,
                            128,
                        );
                    } else {
                        let (kern, sgs): (&'static str, usize) = if m > 1 {
                            let rt = match qw.kern {
                                "linear_quik4" => "linear_quik4_rt",
                                "linear_quik6" => "linear_quik6_rt",
                                "linear_q4k" => "linear_q4k_rt",
                                "linear_q6k" => "linear_q6k_rt",
                                _ => "linear_quik8_rt",
                            };
                            (rt, m.div_ceil(8) * out_f)
                        } else {
                            (qw.kern, out_f)
                        };
                        let pso = self.pipelines.get(kern)?;
                        self.encode_tg(
                            r,
                            &pso,
                            &[bx.as_ref(), &qw.codes, &qw.scm, &qw.dd, bd.as_ref()],
                            &p,
                            sgs * 32,
                            32,
                        );
                    }
                } else {
                    // f16/f32/bf16 weight: dequant-to-f32 device buffer, cached.
                    let bw = self.weight_buf(weight, g, bindings);
                    let pso = self.pipelines.get("linear_f32")?;
                    self.encode_tg(
                        r,
                        &pso,
                        &[bx.as_ref(), bw.as_ref(), bd.as_ref()],
                        &p,
                        m * out_f * 32,
                        32,
                    );
                }
                r.loc[dst.0 as usize] = Loc::Device;
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
                let bx = self.ensure_device(r, x);
                let bpos = self.ensure_device(r, positions);
                let bff = match freq_factors {
                    Some(f) => self.ensure_device(r, f),
                    None => Arc::new(self.zeros_buf(1)),
                };
                let bd = self.dev_dst(r, dst, rows * nh * hd);
                let pso = self.pipelines.get("rope_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&(rope_dim).to_ne_bytes());
                p.extend_from_slice(&theta.to_ne_bytes());
                p.extend_from_slice(&(freq_factors.is_some() as u32).to_ne_bytes());
                self.encode(
                    r,
                    &pso,
                    &[bx.as_ref(), bpos.as_ref(), bff.as_ref(), bd.as_ref()],
                    &p,
                    rows * nh,
                );
                r.loc[dst.0 as usize] = Loc::Device;
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
                let bx = self.ensure_device(r, x);
                let bw = self.weight_buf(weight, g, bindings);
                let bpos = self.ensure_device(r, positions);
                let bff = match freq_factors {
                    Some(f) => self.ensure_device(r, f),
                    None => Arc::new(self.zeros_buf(1)),
                };
                let bd = self.dev_dst(r, dst, rows * nh * hd);
                let pso = self.pipelines.get("qknormrope_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&(rope_dim).to_ne_bytes());
                p.extend_from_slice(&theta.to_ne_bytes());
                p.extend_from_slice(&eps.to_ne_bytes());
                p.extend_from_slice(&(freq_factors.is_some() as u32).to_ne_bytes());
                // One simdgroup per (row, head) — see `qknormrope_f32`.
                self.encode_tg(
                    r,
                    &pso,
                    &[
                        bx.as_ref(),
                        bw.as_ref(),
                        bpos.as_ref(),
                        bff.as_ref(),
                        bd.as_ref(),
                    ],
                    &p,
                    rows * nh * 32,
                    32,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::Add { a, b, dst, n } => {
                let n = n as usize;
                let ba = self.ensure_device(r, a);
                let bb = self.ensure_device(r, b);
                let bd = self.dev_dst(r, dst, n);
                let pso = self.pipelines.get("add_f32")?;
                self.encode(
                    r,
                    &pso,
                    &[ba.as_ref(), bb.as_ref(), bd.as_ref()],
                    &(n as u32).to_ne_bytes(),
                    n,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::Scale { x, dst, s, n } => {
                let n = n as usize;
                let bx = self.ensure_device(r, x);
                let bd = self.dev_dst(r, dst, n);
                let pso = self.pipelines.get("scale_f32")?;
                let mut p = s.to_ne_bytes().to_vec();
                p.extend_from_slice(&(n as u32).to_ne_bytes());
                self.encode(r, &pso, &[bx.as_ref(), bd.as_ref()], &p, n);
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::Softcap { x, dst, cap, n } => {
                let n = n as usize;
                let bx = self.ensure_device(r, x);
                let bd = self.dev_dst(r, dst, n);
                let pso = self.pipelines.get("softcap_f32")?;
                let mut p = cap.to_ne_bytes().to_vec();
                p.extend_from_slice(&(n as u32).to_ne_bytes());
                self.encode(r, &pso, &[bx.as_ref(), bd.as_ref()], &p, n);
                r.loc[dst.0 as usize] = Loc::Device;
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
                let bg = self.ensure_device(r, gate);
                let bu = self.ensure_device(r, up);
                let bd = self.dev_dst(r, dst, rows * nff);
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
                self.encode(
                    r,
                    &pso,
                    &[bg.as_ref(), bu.as_ref(), bd.as_ref()],
                    &p,
                    rows * nff,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::WriteKv {
                src,
                cache,
                rows,
                row_stride,
                pos,
            } => {
                // Stateful cast-copy of `rows` rows into the persistent KV buffer, on the GPU so it
                // stays in the batch (a host write would force a per-layer flush). Metal's hazard
                // tracking orders this write before the Attention that reads the same cache buffer.
                let (rows, rs, pos) = (rows as usize, row_stride as usize, pos as usize);
                let bsrc = self.ensure_device(r, src);
                let cbuf = metal_buf(
                    bindings
                        .get(cache)
                        .expect("metal backend: unbound KV cache"),
                );
                let base = pos * rs;
                let n = rows * rs;
                let kern = match g.desc(cache).dtype {
                    DType::F16 => "writekv_f16",
                    _ => "writekv_f32",
                };
                let pso = self.pipelines.get(kern)?;
                let mut p = (n as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(base as u32).to_ne_bytes());
                self.encode(r, &pso, &[bsrc.as_ref(), &cbuf.raw], &p, n);
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
                // Read K/V straight from the bound cache buffers on-device (no host materialize):
                // f16 caches use the half-typed kernel, f32 caches the plain one. The WriteKv above
                // wrote this same buffer earlier in the batch; hazard tracking makes it visible here.
                let kbuf = metal_buf(bindings.get(k_cache).expect("metal: unbound k_cache"));
                let vbuf = metal_buf(bindings.get(v_cache).expect("metal: unbound v_cache"));
                let bq = self.ensure_device(r, q);
                let bd = self.dev_dst(r, dst, rows * nh * hd);
                let window: u32 = match mask {
                    infr_core::graph::AttnMask::Causal => 0,
                    infr_core::graph::AttnMask::SlidingWindow(w) => w as u32,
                };
                // Route by kernel-latency shape. The one-simdgroup-per-(query, head) kernels are
                // latency-bound on their serial O(kv_len) online-softmax chain, so any long
                // context takes a split-KV kernel — NSG simdgroups per (query, head) merged in
                // threadgroup memory — 32-way when head_dim fits its threadgroup accumulator
                // (hd <= 128), 8-way otherwise. That holds for prefill too (measured, not just
                // decode): the extra parallelism is redundant there, but the 4x-8x shorter serial
                // chain still wins. Short contexts keep the lean unsplit kernel. (A
                // simdgroup_matrix flash-attention kernel was built and benched here; it never
                // beat the 32-way split on prefill — occupancy-starved by its threadgroup tiles —
                // so it was dropped.)
                let f16 = g.desc(k_cache).dtype == DType::F16;
                // Wide launches on an f16 cache take the half-fragment flash kernel: K^T/V
                // fragments load straight from the cache, Q is cast once to f16 below. Small
                // kv_len stays scalar (also keeps the short-kv wide parity test on the exact
                // path). See `attnflash_f16kv` for why the f32 flash attempt lost and this wins.
                let flash = f16 && rows * nh >= 128 && kv_len >= 64 && hd <= 128 && hd % 8 == 0;
                let split = !flash && (rows * nh < 128 || kv_len >= 128);
                // The split kernels REQUIRE their full NSG*32-thread threadgroup: every simdgroup
                // owns a strided KV slice, so a smaller launch would silently skip positions and
                // merge uninitialized partials. maxTotalThreadsPerThreadgroup is per-PIPELINE
                // (register pressure or a paravirtual device can push it below 1024 — GitHub's
                // macOS CI runners do), so each width is gated on its own pipeline's cap and
                // degrades to the next-narrower kernel instead of letting `encode_tg` clamp.
                let fits = |name: &'static str, threads: u64| -> Result<bool> {
                    Ok(self
                        .pipelines
                        .get(name)?
                        .max_total_threads_per_threadgroup()
                        >= threads)
                };
                let split32 = split
                    && kv_len >= 128
                    && hd <= 128
                    && fits(
                        if f16 {
                            "attnsplit32_f16kv"
                        } else {
                            "attnsplit32_f32"
                        },
                        1024,
                    )?;
                let split = split
                    && (split32
                        || fits(
                            if f16 {
                                "attnsplit_f16kv"
                            } else {
                                "attnsplit_f32"
                            },
                            256,
                        )?);
                let kern = match (flash, f16, split, split32) {
                    (true, ..) => "attnflash_f16kv",
                    (_, true, false, _) => "attention_f16kv",
                    (_, true, _, true) => "attnsplit32_f16kv",
                    (_, true, true, false) => "attnsplit_f16kv",
                    (_, false, false, _) => "attention_f32",
                    (_, false, _, true) => "attnsplit32_f32",
                    (_, false, true, false) => "attnsplit_f32",
                };
                let pso = self.pipelines.get(kern)?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(kv_len as u32).to_ne_bytes());
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(nkv as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&scale.to_ne_bytes());
                p.extend_from_slice(&window.to_ne_bytes());
                p.extend_from_slice(&pos.to_ne_bytes());
                if flash {
                    // Cast q to a transient f16 buffer (one pass), then one flash dispatch:
                    // one simdgroup per (8-query tile, head).
                    let n = rows * nh * hd;
                    let qh =
                        Arc::new(self.device.new_buffer(
                            (n * 2).max(4) as u64,
                            MTLResourceOptions::StorageModeShared,
                        ));
                    let cast = self.pipelines.get("cast_f32_f16")?;
                    self.encode(r, &cast, &[bq.as_ref(), &qh], &(n as u32).to_ne_bytes(), n);
                    self.encode_tg(
                        r,
                        &pso,
                        &[qh.as_ref(), &kbuf.raw, &vbuf.raw, bd.as_ref()],
                        &p,
                        rows.div_ceil(8) * nh * 32,
                        32,
                    );
                } else {
                    // One simdgroup per (query, head); split kernels use NSG simdgroups per
                    // pair, grid still exactly rows*n_head threadgroups.
                    let nsg = if split32 {
                        32
                    } else if split {
                        8
                    } else {
                        1
                    };
                    self.encode_tg(
                        r,
                        &pso,
                        &[bq.as_ref(), &kbuf.raw, &vbuf.raw, bd.as_ref()],
                        &p,
                        rows * nh * nsg * 32,
                        nsg * 32,
                    );
                }
                r.loc[dst.0 as usize] = Loc::Device;
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
                self.ensure_host(r, g, x);
                let xs = r.vals[x.0 as usize].clone();
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
                r.vals[dst.0 as usize] = out;
                r.loc[dst.0 as usize] = Loc::Host;
            }
            // Sequential per-token recurrences (control-flow heavy, tiny) — host-side, exactly
            // mirroring the CPU reference. The recurrent `state` is a bound f32 Input read/written
            // through `vals` (loaded in the preamble, written back after the op loop).
            Op::Conv1dSilu {
                x,
                weight,
                state,
                dst,
                rows,
                channels,
                kernel,
            } => {
                let (rr, cc, kk) = (rows as usize, channels as usize, kernel as usize);
                self.ensure_host(r, g, x);
                self.ensure_host(r, g, state);
                let xs = r.vals[x.0 as usize].clone(); // [rows, channels]
                let ws = self.weight_host(weight, g, bindings); // [channels, kernel]
                let st = &mut r.vals[state.0 as usize]; // [(kernel-1), channels], oldest first
                let mut out = vec![0f32; rr * cc];
                for t in 0..rr {
                    let xt = &xs[t * cc..t * cc + cc];
                    for ch in 0..cc {
                        let mut acc = 0f32;
                        for j in 0..kk - 1 {
                            acc += st[j * cc + ch] * ws[ch * kk + j];
                        }
                        acc += xt[ch] * ws[ch * kk + (kk - 1)];
                        out[t * cc + ch] = acc / (1.0 + (-acc).exp()); // silu
                    }
                    for j in 0..kk.saturating_sub(2) {
                        for ch in 0..cc {
                            st[j * cc + ch] = st[(j + 1) * cc + ch];
                        }
                    }
                    if kk >= 2 {
                        for ch in 0..cc {
                            st[(kk - 2) * cc + ch] = xt[ch];
                        }
                    }
                }
                r.vals[dst.0 as usize] = out;
                r.loc[dst.0 as usize] = Loc::Host;
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
                rows,
                n_vhead,
                n_khead,
                head_k,
                head_v,
                eps,
            } => {
                let (rr, nv, nk, kd, vd) = (
                    rows as usize,
                    n_vhead as usize,
                    n_khead as usize,
                    head_k as usize,
                    head_v as usize,
                );
                for id in [q, k, v, b, a, state] {
                    self.ensure_host(r, g, id);
                }
                let qf = r.vals[q.0 as usize].clone(); // [rows, nk*kd]
                let kf = r.vals[k.0 as usize].clone();
                let vf = r.vals[v.0 as usize].clone(); // [rows, nv*vd]
                let bf = r.vals[b.0 as usize].clone(); // [rows, nv]
                let af = r.vals[a.0 as usize].clone();
                let acoef = self.weight_host(a_coef, g, bindings);
                let dtb = self.weight_host(dt_bias, g, bindings);
                let st = &mut r.vals[state.0 as usize]; // [nv, kd, vd]
                let mut out = vec![0f32; rr * nv * vd];
                let qscale = 1.0 / (kd as f32).sqrt();
                let l2 = |slice: &[f32]| -> f32 {
                    (slice.iter().map(|x| x * x).sum::<f32>() + eps).sqrt()
                };
                // Sequential scan over rows, carrying the per-head state S across tokens.
                for t in 0..rr {
                    let (qb, vb, bb) = (t * nk * kd, t * nv * vd, t * nv);
                    for h in 0..nv {
                        let kh_idx = h % nk; // GQA: q/k heads tiled to nv value heads
                        let mut qh = qf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
                        let mut kh = kf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
                        let vh = &vf[vb + h * vd..vb + h * vd + vd];
                        let qn = l2(&qh);
                        let kn = l2(&kh);
                        for x in qh.iter_mut() {
                            *x = *x / qn * qscale;
                        }
                        for x in kh.iter_mut() {
                            *x /= kn;
                        }
                        let beta = 1.0 / (1.0 + (-bf[bb + h]).exp());
                        let sp = {
                            let z = af[bb + h] + dtb[h];
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
                        let oh = &mut out[vb + h * vd..vb + h * vd + vd];
                        for kk in 0..kd {
                            let qv = qh[kk];
                            let row = &sh[kk * vd..kk * vd + vd];
                            for d in 0..vd {
                                oh[d] += qv * row[d];
                            }
                        }
                    }
                }
                r.vals[dst.0 as usize] = out;
                r.loc[dst.0 as usize] = Loc::Host;
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
                self.ensure_host(r, g, src);
                self.ensure_host(r, g, dst);
                let s = r.vals[src.0 as usize].clone();
                r.vals[dst.0 as usize][dof..dof + n].copy_from_slice(&s[so..so + n]);
                r.loc[dst.0 as usize] = Loc::Host;
            }
            Op::CopyStrided {
                src,
                src_off,
                src_stride,
                dst,
                dst_off,
                dst_stride,
                rows,
                n,
            } => {
                let (so, ss, dof, ds, n) = (
                    src_off as usize,
                    src_stride as usize,
                    dst_off as usize,
                    dst_stride as usize,
                    n as usize,
                );
                self.ensure_host(r, g, src);
                self.ensure_host(r, g, dst);
                let s = r.vals[src.0 as usize].clone();
                let d = &mut r.vals[dst.0 as usize];
                for rr in 0..rows as usize {
                    d[dof + rr * ds..dof + rr * ds + n]
                        .copy_from_slice(&s[so + rr * ss..so + rr * ss + n]);
                }
                r.loc[dst.0 as usize] = Loc::Host;
            }
        }
        Ok(())
    }
}
