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
/// so they share a single commit + wait instead of one per op. When `tape` is armed, every encoded
/// dispatch is also recorded for decode replay (see [`Tape`]), and `posbuf` carries the bound
/// positions buffer the dynamic-pos kernels read.
struct Resident {
    vals: Vec<Vec<f32>>,
    dev: Vec<Option<Arc<MtlBuffer>>>,
    loc: Vec<Loc>,
    cb: Option<CommandBuffer>,
    enc: Option<ComputeCommandEncoder>,
    tape: Option<Vec<TapeEntry>>,
    posbuf: Option<MtlBuffer>,
    /// Linear→Add residual fusion (see `linear_add_peephole`): Linear op index → (residual
    /// tensor, final dst); the absorbed Adds are in `skip`.
    fused: std::collections::HashMap<usize, (TensorId, TensorId)>,
    skip: std::collections::HashSet<usize>,
    /// Host-read override of the graph's baked position (see `run_graph`): the seam's replay
    /// loop re-executes one compiled decode graph (baked pos=0) and only rewrites the bound
    /// positions buffer, so on any decode-shaped graph the position-consuming ops (Attention,
    /// WriteKv) must take the CURRENT position — the tape path reads it on the GPU; this covers
    /// every graph the tape can't (MoE, gemma shapes, hd without a dyn instantiation).
    dynpos: Option<u32>,
}

/// Fuse `Linear (m==1, Q4_K/Q6_K weight, Internal dst) → Add(residual)` into the fused-residual
/// GEMV (`linear_q4k_add`/`linear_q6k_add`) — one dispatch + one dependency stage instead of two,
/// and no round-trip of the sublayer output (the decode o_proj/down_proj shape; mirrors the
/// Vulkan adapter's peephole). Only the IMMEDIATELY following Add fuses: the seam builder emits
/// the pair adjacent for non-gemma models, and gemma's sandwich norm sits between and correctly
/// blocks it.
fn linear_add_peephole(
    g: &infr_core::graph::Graph,
) -> (
    std::collections::HashMap<usize, (TensorId, TensorId)>,
    std::collections::HashSet<usize>,
) {
    let mut fused = std::collections::HashMap::new();
    let mut skip = std::collections::HashSet::new();
    for (i, op) in g.ops.iter().enumerate() {
        let Op::Linear {
            dst, m: 1, weight, ..
        } = op
        else {
            continue;
        };
        if !matches!(g.tensors[dst.0 as usize].kind, TensorKind::Internal) {
            continue;
        }
        if !matches!(g.desc(*weight).dtype, DType::Q4K | DType::Q6K) {
            continue;
        }
        if let Some(Op::Add {
            a, b, dst: add_dst, ..
        }) = g.ops.get(i + 1)
        {
            let residual = if a == dst {
                *b
            } else if b == dst {
                *a
            } else {
                continue;
            };
            fused.insert(i, (residual, *add_dst));
            skip.insert(i + 1);
        }
    }
    (fused, skip)
}

/// One recorded dispatch: everything `encode_tg` needs to re-encode it verbatim. The buffer clones
/// are retains, so the tape keeps every transient (intermediate activations, cached weights) alive
/// across replays — replaying reuses the exact buffers the recording ran on.
pub(crate) struct TapeEntry {
    pso: ComputePipelineState,
    bufs: Vec<MtlBuffer>,
    /// Per-buffer byte offset for `set_buffer` (parallel to `bufs`; all zero except the
    /// fused-QKV weight slices — see the Linear arm's `w_off`).
    offs: Vec<u64>,
    params: Vec<u8>,
    threads: usize,
    tg: usize,
}

/// A recorded decode forward for the seam's record-once replay (`Capabilities::decode_replay`):
/// the engine compiles the decode graph once (baked pos=0), binds everything once, and re-executes
/// the same plan per token after uploading the new embedding + position. Metal command buffers are
/// single-use, so "replay" here is re-encoding this flat dispatch list — no graph walk, no host
/// mirror, no transient allocation, no routing. Ops whose behavior depends on the position use
/// DYNAMIC-POS kernels that read the bound positions buffer (`attnvec_dyn_*`, `writekv_dyn_f16`;
/// RoPE already reads it), so the recorded params stay valid for every token.
pub(crate) struct Tape {
    /// Fingerprint of (op discriminants, tensor count, bound buffer addresses): a tape is replayed
    /// only for a graph with the identical op sequence over the identical bindings. Baked pos /
    /// kv_len MAY differ across tokens — the dynamic-pos kernels ignore them by construction.
    fp: u64,
    entries: Vec<TapeEntry>,
}

/// Fingerprint the (graph shape, bindings) pair for tape matching. Op kind + tensor count pins the
/// graph structure; the bound buffer addresses pin the weights/caches/IO. Two graphs that agree on
/// all of that and pass [`replay_shape`] differ at most in baked positions, which replay reads
/// dynamically.
fn replay_fp(g: &infr_core::graph::Graph, bindings: &Bindings) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x100000001b3);
    };
    mix(g.tensors.len() as u64);
    for op in &g.ops {
        mix(op_name(op).as_ptr() as u64);
    }
    for i in 0..g.tensors.len() {
        if let Some(b) = bindings.get(TensorId(i as u32)) {
            mix(metal_buf(b) as *const _ as u64);
        }
    }
    h
}

/// Is this graph the decode shape the replay tape supports? Every op must be one the recorder
/// handles fully on-device, attention must be the rows=1 f16 shape with a dynamic-pos kernel
/// instantiation (hd 64/128), and a QkNormRope must exist to name the positions buffer.
fn replay_shape(g: &infr_core::graph::Graph, bindings: &Bindings) -> bool {
    use infr_core::graph::TensorKind;
    let mut has_rope = false;
    let mut has_attn = false;
    for op in &g.ops {
        match op {
            Op::RmsNorm { .. }
            | Op::Linear { .. }
            | Op::GatedAct { .. }
            | Op::GatedActFused { .. }
            | Op::Add { .. } => {}
            Op::QkNormRope { .. } => has_rope = true,
            Op::WriteKv { cache, .. } => {
                if !matches!(g.desc(*cache).dtype, DType::F16 | DType::Q8_0) {
                    return false;
                }
            }
            Op::Attention {
                rows,
                head_dim,
                k_cache,
                ..
            } => {
                has_attn = true;
                if *rows != 1
                    || !matches!(*head_dim, 64 | 128)
                    || !matches!(g.desc(*k_cache).dtype, DType::F16 | DType::Q8_0)
                {
                    return false;
                }
            }
            _ => return false,
        }
    }
    // Every non-in-place tensor the recording direct-binds must actually be bound.
    for (i, decl) in g.tensors.iter().enumerate() {
        let needs_binding = matches!(decl.kind, TensorKind::Input | TensorKind::Output);
        if needs_binding && bindings.get(TensorId(i as u32)).is_none() {
            return false;
        }
    }
    has_rope && has_attn
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
        Op::GatedActFused { .. } => "GatedActFused",
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

    /// A reusable op-scratch buffer (see `MetalBackend::scratch`): get-or-create by
    /// (f32 count, tag).
    fn scratch_buf(&self, n: usize, tag: u8) -> Arc<MtlBuffer> {
        self.scratch
            .lock()
            .unwrap()
            .entry((n, tag))
            .or_insert_with(|| Arc::new(self.zeros_buf(n)))
            .clone()
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
        let with_offs: Vec<(&MtlBuffer, u64)> = bufs.iter().map(|b| (*b, 0)).collect();
        self.encode_tg_off(r, pso, &with_offs, params, threads, tg);
    }

    /// As `encode_tg`, but each buffer binds at a byte offset — the fused-QKV Linear slices
    /// bind the shared concatenated weight at each projection's row offset (`Op::Linear.w_off`).
    fn encode_tg_off(
        &self,
        r: &mut Resident,
        pso: &ComputePipelineState,
        bufs: &[(&MtlBuffer, u64)],
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
        for (i, (b, off)) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), *off);
        }
        if !params.is_empty() {
            enc.set_bytes(
                bufs.len() as u64,
                params.len() as u64,
                params.as_ptr() as *const c_void,
            );
        }
        let cap = pso.max_total_threads_per_threadgroup() as usize;
        let tgw = if tg == 0 { cap } else { tg.min(cap) }.min(threads.max(1)) as u64;
        enc.dispatch_threads(MTLSize::new(threads as u64, 1, 1), MTLSize::new(tgw, 1, 1));
        if let Some(tape) = r.tape.as_mut() {
            tape.push(TapeEntry {
                pso: pso.clone(),
                bufs: bufs.iter().map(|(b, _)| (*b).clone()).collect(),
                offs: bufs.iter().map(|(_, off)| *off).collect(),
                params: params.to_vec(),
                threads,
                tg,
            });
        }
    }

    /// Re-encode a recorded tape: one command buffer, the flat dispatch list, commit + wait.
    /// This IS the per-token decode cost on the replay path — no graph walk, no host mirror,
    /// no allocation.
    fn replay_tape(&self, tape: &Tape) {
        objc::rc::autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for e in &tape.entries {
                enc.set_compute_pipeline_state(&e.pso);
                for (i, b) in e.bufs.iter().enumerate() {
                    enc.set_buffer(i as u64, Some(b), e.offs[i]);
                }
                if !e.params.is_empty() {
                    enc.set_bytes(
                        e.bufs.len() as u64,
                        e.params.len() as u64,
                        e.params.as_ptr() as *const c_void,
                    );
                }
                let cap = e.pso.max_total_threads_per_threadgroup() as usize;
                let tg = if e.tg == 0 { cap } else { e.tg.min(cap) }.min(e.threads.max(1)) as u64;
                enc.dispatch_threads(MTLSize::new(e.threads as u64, 1, 1), MTLSize::new(tg, 1, 1));
            }
            enc.end_encoding();
            let t0 = self.profiling.then(std::time::Instant::now);
            cb.commit();
            cb.wait_until_completed();
            if let Some(t0) = t0 {
                let mut pr = self.prof.lock().unwrap();
                pr.add_dispatch(t0.elapsed());
                pr.add_forward();
            }
        });
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

        // Decode replay: if this exact (graph shape, bindings) pair was recorded, re-encode the
        // tape and skip the graph walk entirely (see `Tape`). The engine's replay loop re-executes
        // one compiled plan with stable bindings, so after the first recorded token every
        // subsequent token takes this path. The dynamic-pos vector kernel REQUIRES its full
        // 1024-thread threadgroup (same silent-clamp hazard as the split kernels), so recording is
        // gated on its pipeline cap — a capped device (CI paravirtual) keeps the per-token path.
        let dyn_cap_ok = || -> bool {
            let kern = g.ops.iter().find_map(|op| match op {
                Op::Attention {
                    head_dim: hd @ (64 | 128),
                    k_cache,
                    ..
                } => Some(match (g.desc(*k_cache).dtype == DType::Q8_0, hd) {
                    (false, 64) => "attnvec_dyn_f16kv_hd64",
                    (false, _) => "attnvec_dyn_f16kv_hd128",
                    (true, 64) => "attnvec_dyn_q8kv_hd64",
                    (true, _) => "attnvec_dyn_q8kv_hd128",
                }),
                _ => None,
            });
            kern.is_some_and(|kn| {
                self.pipelines
                    .get(kn)
                    .map(|pl| pl.max_total_threads_per_threadgroup() >= 1024)
                    .unwrap_or(false)
            })
        };
        if replay_shape(g, bindings) && dyn_cap_ok() {
            let fp = replay_fp(g, bindings);
            if let Some(tape) = self.replay.lock().unwrap().as_ref() {
                if tape.fp == fp {
                    self.replay_tape(tape);
                    return Ok(());
                }
            }
            let entries = objc::rc::autoreleasepool(|| self.run_graph(g, bindings, true))?;
            if let Some(entries) = entries {
                *self.replay.lock().unwrap() = Some(Tape { fp, entries });
            }
            return Ok(());
        }

        // Wrap the whole forward in one autorelease pool: the batched command buffers/encoders are
        // retained owned handles, so we drain the pool once per forward instead of once per op.
        objc::rc::autoreleasepool(|| self.run_graph(g, bindings, false).map(|_| ()))
    }

    fn run_graph(
        &self,
        g: &infr_core::graph::Graph,
        bindings: &Bindings,
        record: bool,
    ) -> Result<Option<Vec<TapeEntry>>> {
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
            tape: record.then(Vec::new),
            posbuf: None,
            fused: std::collections::HashMap::new(),
            skip: std::collections::HashSet::new(),
            dynpos: None,
        };
        // Decode-shaped graph (a rows==1 Attention) with a bound positions buffer: read the
        // CURRENT position off the shared-memory buffer and override the baked pos/kv_len in
        // the position-consuming arms. Single-execute callers bind positions equal to the baked
        // value, so this is the identity for them; the seam's replay loop is where they differ.
        let decode_shaped = g
            .ops
            .iter()
            .any(|op| matches!(op, Op::Attention { rows: 1, .. }));
        if decode_shaped {
            let positions = g.ops.iter().find_map(|op| match op {
                Op::QkNormRope { positions, .. } | Op::Rope { positions, .. } => Some(*positions),
                _ => None,
            });
            if let Some(pid) = positions {
                if let Some(b) = bindings.get(pid) {
                    let buf = metal_buf(b);
                    if buf.len >= 4 {
                        let v = unsafe { *(buf.raw.contents() as *const i32) };
                        if v >= 0 {
                            r.dynpos = Some(v as u32);
                        }
                    }
                }
            }
        }
        {
            let (fused, skip) = linear_add_peephole(g);
            r.fused = fused;
            r.skip = skip;
        }
        if record {
            // Recording (`replay_shape` held): the tape must read/write the BOUND buffers, not
            // per-execute host-mirror copies — the engine mutates the bound hidden/positions
            // buffers between replays and reads logits from its bound buffer. So f32 Inputs and
            // Outputs are direct-bound as the tensor's device buffer, and the positions buffer
            // (i32, named by the graph's QkNormRope) is stashed for the dynamic-pos kernels.
            for (i, decl) in g.tensors.iter().enumerate() {
                let id = TensorId(i as u32);
                if direct.contains(&id) {
                    continue;
                }
                let bound = match decl.kind {
                    TensorKind::Input if decl.desc.dtype == DType::F32 => true,
                    TensorKind::Output => true,
                    _ => false,
                };
                if bound {
                    let buf = metal_buf(bindings.get(id).expect("replay_shape checked bindings"));
                    r.dev[i] = Some(Arc::new(buf.raw.clone()));
                    if decl.kind == TensorKind::Input {
                        r.loc[i] = Loc::Device;
                    }
                }
            }
            let positions = g
                .ops
                .iter()
                .find_map(|op| match op {
                    Op::QkNormRope { positions, .. } => Some(*positions),
                    _ => None,
                })
                .expect("replay_shape checked QkNormRope");
            r.posbuf = Some(metal_buf(bindings.get(positions).unwrap()).raw.clone());
        }
        for (i, decl) in g.tensors.iter().enumerate() {
            match decl.kind {
                TensorKind::Internal | TensorKind::Output => {
                    r.vals[i] = vec![0f32; decl.desc.numel()]
                }
                TensorKind::Input if direct.contains(&TensorId(i as u32)) => {}
                TensorKind::Input if record && decl.desc.dtype == DType::F32 => {}
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
            for (idx, op) in g.ops.iter().enumerate() {
                if r.skip.contains(&idx) {
                    continue;
                }
                let t0 = std::time::Instant::now();
                self.run_op(op, idx, g, bindings, &mut r)?;
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
            for (idx, op) in g.ops.iter().enumerate() {
                if r.skip.contains(&idx) {
                    continue;
                }
                self.run_op(op, idx, g, bindings, &mut r)?;
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
                let b = metal_buf(bindings.get(TensorId(i as u32)).unwrap());
                // Direct-bound (recording): the GPU already wrote the bound buffer — nothing to
                // copy, the trailing flush below makes it visible.
                if matches!(&r.dev[i], Some(d) if d.contents() == b.raw.contents()) {
                    continue;
                }
                // Pull the value back to the host mirror (flushes the batch if it's still on-device).
                self.ensure_host(&mut r, g, TensorId(i as u32));
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
        Ok(r.tape.take())
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
        idx: usize,
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
                w_off,
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
                    // Fused-QKV slice: `w_off` is a whole-row element offset into the shared
                    // concatenated weight (the runner bakes it as `row0 * in_f`), so it lands on
                    // a block boundary in every stream — bind each stream at the corresponding
                    // byte offset and the kernels see the slice as a standalone [out_f, in_f]
                    // weight. Native K-quant streams are raw GGUF blocks (144 B / 210 B per 256
                    // elements); factored streams are bit-packed codes + 4 B scm per 16 elements
                    // + 4 B (d, dmin) per 2^dshift elements.
                    let e = w_off as u64;
                    let dd_off = (e >> qw.dshift) * 4;
                    let (codes_off, scm_off, dd_off) = match qw.kern {
                        _ if w_off == 0 => (0, 0, 0),
                        // Native kernels read raw GGUF blocks; scm/dd are dummy buffers.
                        "linear_q4k" => (e / 256 * 144, 0, 0),
                        "linear_q6k" => (e / 256 * 210, 0, 0),
                        "linear_quik4" => (e / 2, e / 4, dd_off),
                        "linear_quik6" => (e / 4 * 3, e / 4, dd_off),
                        _ => (e, e / 4, dd_off),
                    };
                    // `set_buffer` offsets must stay 4-byte aligned; every real fused shape does
                    // (in_f is a multiple of 512 or the format's stride is already 4-aligned) —
                    // reject the pathological remainder instead of tripping API validation.
                    if codes_off % 4 != 0 || scm_off % 4 != 0 || dd_off % 4 != 0 {
                        return Err(Error::Unsupported(
                            "metal: Linear w_off lands off 4-byte alignment".into(),
                        ));
                    }
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
                    let cmm_kern = match qw.kern {
                        "linear_quik4" => "linear_quik4_cmm",
                        "linear_quik6" => "linear_quik6_cmm",
                        "linear_q4k" => "linear_q4k_cmm",
                        "linear_q6k" => "linear_q6k_cmm",
                        _ => "linear_quik8_cmm",
                    };
                    // Prefer the cooperative 32x64 threadgroup tile; per-simdgroup HGEMM covers
                    // the out_f % 64 != 0 leftovers, and both need the full 128-thread group.
                    let cmm_ok = m >= 16
                        && out_f % 64 == 0
                        && self
                            .pipelines
                            .get(cmm_kern)?
                            .max_total_threads_per_threadgroup()
                            >= 128;
                    let hmm_ok = m >= 16
                        && out_f % 16 == 0
                        && self
                            .pipelines
                            .get(hmm_kern)?
                            .max_total_threads_per_threadgroup()
                            >= 128;
                    p.extend_from_slice(&qw.dshift.to_ne_bytes());
                    if cmm_ok {
                        // Reads f32 x directly (casts to f16 while staging) — no cast pass.
                        let pso = self.pipelines.get(cmm_kern)?;
                        self.encode_tg_off(
                            r,
                            &pso,
                            &[
                                (bx.as_ref(), 0),
                                (&qw.codes, codes_off),
                                (&qw.scm, scm_off),
                                (&qw.dd, dd_off),
                                (bd.as_ref(), 0),
                            ],
                            &p,
                            m.div_ceil(32) * (out_f / 64) * 128,
                            128,
                        );
                    } else if hmm_ok {
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
                        self.encode_tg_off(
                            r,
                            &pso,
                            &[
                                (xh.as_ref(), 0),
                                (&qw.codes, codes_off),
                                (&qw.scm, scm_off),
                                (&qw.dd, dd_off),
                                (bd.as_ref(), 0),
                            ],
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
                            // The native mul_mv-shape GEMVs cover TWO output rows per simdgroup.
                            let sgs = match qw.kern {
                                "linear_q4k" | "linear_q6k" => out_f.div_ceil(2),
                                _ => out_f,
                            };
                            (qw.kern, sgs)
                        };
                        // Residual peephole: this Linear absorbed the following Add — take the
                        // fused-residual variant and write the Add's dst directly.
                        if let Some(&(res, fdst)) = r.fused.get(&idx) {
                            let fk = match kern {
                                "linear_q4k" => "linear_q4k_add",
                                _ => "linear_q6k_add",
                            };
                            let bres = self.ensure_device(r, res);
                            let bfd = self.dev_dst(r, fdst, out_f);
                            let pso = self.pipelines.get(fk)?;
                            self.encode_tg_off(
                                r,
                                &pso,
                                &[
                                    (bx.as_ref(), 0),
                                    (&qw.codes, codes_off),
                                    (&qw.scm, scm_off),
                                    (&qw.dd, dd_off),
                                    (bfd.as_ref(), 0),
                                    (bres.as_ref(), 0),
                                ],
                                &p,
                                sgs * 32,
                                32,
                            );
                            r.loc[fdst.0 as usize] = Loc::Device;
                            return Ok(());
                        }
                        let pso = self.pipelines.get(kern)?;
                        self.encode_tg_off(
                            r,
                            &pso,
                            &[
                                (bx.as_ref(), 0),
                                (&qw.codes, codes_off),
                                (&qw.scm, scm_off),
                                (&qw.dd, dd_off),
                                (bd.as_ref(), 0),
                            ],
                            &p,
                            sgs * 32,
                            32,
                        );
                    }
                } else {
                    // f16/f32/bf16 weight: dequant-to-f32 device buffer, cached.
                    let bw = self.weight_buf(weight, g, bindings);
                    let pso = self.pipelines.get("linear_f32")?;
                    self.encode_tg_off(
                        r,
                        &pso,
                        &[
                            (bx.as_ref(), 0),
                            (bw.as_ref(), w_off as u64 * 4),
                            (bd.as_ref(), 0),
                        ],
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
                // The kernel reads the bound i32 positions buffer directly (exact widening in
                // the shader) — no host round-trip, and the decode-replay tape stays valid when
                // the engine rewrites the position between replays.
                let bpos = Arc::new(
                    metal_buf(
                        bindings
                            .get(positions)
                            .expect("metal backend: unbound positions"),
                    )
                    .raw
                    .clone(),
                );
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
            // Only produced when `Capabilities::combined_gu` is set — Metal leaves it false (the
            // reference backend keeps the separate gate/up form), so this arm is unreachable.
            Op::GatedActFused {
                gu,
                dst,
                rows,
                nff,
                act,
            } => {
                let (rows, nff) = (rows as usize, nff as usize);
                let bg = self.ensure_device(r, gu);
                let bd = self.dev_dst(r, dst, rows * nff);
                let pso = self.pipelines.get("gatedactfused_f32")?;
                let act_code: u32 = match act {
                    infr_core::graph::Activation::Silu => 0,
                    infr_core::graph::Activation::Gelu => 1,
                    infr_core::graph::Activation::Sigmoid => 2,
                };
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nff as u32).to_ne_bytes());
                p.extend_from_slice(&act_code.to_ne_bytes());
                p.extend_from_slice(&0u32.to_ne_bytes()); // pad to GatedParams
                self.encode(r, &pso, &[bg.as_ref(), bd.as_ref()], &p, rows * nff);
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
                let (rows, rs, mut pos) = (rows as usize, row_stride as usize, pos as usize);
                if let (Some(dp), 1) = (r.dynpos, rows) {
                    pos = dp as usize;
                }
                let bsrc = self.ensure_device(r, src);
                let cbuf = metal_buf(
                    bindings
                        .get(cache)
                        .expect("metal backend: unbound KV cache"),
                );
                let base = pos * rs;
                let n = rows * rs;
                let q8 = g.desc(cache).dtype == DType::Q8_0;
                if let Some(posbuf) = r.posbuf.clone() {
                    // Recording a replay tape: the row offset must come from the positions buffer
                    // (the baked `pos` is this token's only) — f16/q8 cache guaranteed by the gate.
                    let pso = self.pipelines.get(if q8 {
                        "writekv_dyn_q8"
                    } else {
                        "writekv_dyn_f16"
                    })?;
                    let mut p = (n as u32).to_ne_bytes().to_vec();
                    p.extend_from_slice(&(rs as u32).to_ne_bytes());
                    let threads = if q8 { n / 32 } else { n };
                    self.encode(r, &pso, &[bsrc.as_ref(), &cbuf.raw, &posbuf], &p, threads);
                } else {
                    let kern = match g.desc(cache).dtype {
                        DType::Q8_0 => "writekv_q8",
                        DType::F16 => "writekv_f16",
                        _ => "writekv_f32",
                    };
                    let pso = self.pipelines.get(kern)?;
                    let mut p = (n as u32).to_ne_bytes().to_vec();
                    p.extend_from_slice(&(base as u32).to_ne_bytes());
                    let threads = if q8 { n / 32 } else { n };
                    self.encode(r, &pso, &[bsrc.as_ref(), &cbuf.raw], &p, threads);
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
                // Replayed decode graph: the baked pos/kv_len are the compile-time token's;
                // take the current position (also steers the kv_len-based kernel routing).
                let (pos, kv_len) = match r.dynpos {
                    Some(dp) if rows == 1 => (dp, dp as usize + 1),
                    _ => (pos, kv_len),
                };
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
                if let Some(posbuf) = r.posbuf.clone() {
                    // Recording a replay tape (rows==1, f16/q8 cache, hd 64/128 — checked by the
                    // gate): route straight to the dynamic-pos vector flash kernel, which reads
                    // pos from the positions buffer and covers every kv_len from 1 up. The usual
                    // shape routing below would bake this token's kv_len into the kernel CHOICE,
                    // which is exactly what a replayed tape can't have.
                    let kern = match (g.desc(k_cache).dtype == DType::Q8_0, hd) {
                        (false, 64) => "attnvec_dyn_f16kv_hd64",
                        (false, _) => "attnvec_dyn_f16kv_hd128",
                        (true, 64) => "attnvec_dyn_q8kv_hd64",
                        (true, _) => "attnvec_dyn_q8kv_hd128",
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
                    self.encode_tg(
                        r,
                        &pso,
                        &[bq.as_ref(), &kbuf.raw, &vbuf.raw, bd.as_ref(), &posbuf],
                        &p,
                        rows * nh * 32 * 32,
                        32 * 32,
                    );
                    r.loc[dst.0 as usize] = Loc::Device;
                    return Ok(());
                }
                // Q8_0 cache: rows==1 decode takes the q8 vector kernel (hd 64/128); every
                // other shape the scalar dequant-on-read fallback (prefill lives here until a
                // q8 flash port). Both dequantize exactly — reassociation-only vs the f16 math.
                if g.desc(k_cache).dtype == DType::Q8_0 {
                    // Prefill-wide launches take the q8 cooperative flash (dequant-staged KV
                    // tiles) — the same gate shape as the f16 flash.
                    let fq8 = rows * nh >= 128 && kv_len >= 64 && matches!(hd, 64 | 128) && {
                        let kn = if hd == 64 {
                            "attnflash2_q8kv_hd64"
                        } else {
                            "attnflash2_q8kv_hd128"
                        };
                        self.pipelines
                            .get(kn)
                            .map(|pl| pl.max_total_threads_per_threadgroup() >= 128)
                            .unwrap_or(false)
                    };
                    if fq8 {
                        let kern = if hd == 64 {
                            "attnflash2_q8kv_hd64"
                        } else {
                            "attnflash2_q8kv_hd128"
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
                        let n = rows * nh * hd;
                        let qh = Arc::new(self.device.new_buffer(
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
                            rows.div_ceil(8) * nh * 128,
                            128,
                        );
                        r.loc[dst.0 as usize] = Loc::Device;
                        return Ok(());
                    }
                    // The vector kernel runs one threadgroup per (row, head), covering decode
                    // and any prefill shape the flash gate declined.
                    let vq8 = matches!(hd, 64 | 128) && kv_len >= 128 && {
                        let kn = if hd == 64 {
                            "attnvec_q8kv_hd64"
                        } else {
                            "attnvec_q8kv_hd128"
                        };
                        self.pipelines
                            .get(kn)
                            .map(|pl| pl.max_total_threads_per_threadgroup() >= 1024)
                            .unwrap_or(false)
                    };
                    let kern = if vq8 {
                        if hd == 64 {
                            "attnvec_q8kv_hd64"
                        } else {
                            "attnvec_q8kv_hd128"
                        }
                    } else {
                        "attention_q8kv"
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
                    let nsg = if vq8 { 32 } else { 1 };
                    self.encode_tg(
                        r,
                        &pso,
                        &[bq.as_ref(), &kbuf.raw, &vbuf.raw, bd.as_ref()],
                        &p,
                        rows * nh * nsg * 32,
                        nsg * 32,
                    );
                    r.loc[dst.0 as usize] = Loc::Device;
                    return Ok(());
                }
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
                // The cooperative flash kernel (4 simdgroups per query tile, llama.cpp
                // flash_attn_ext structure) is instantiated per compile-time head size
                // (fully unrolled QK/PV loops) and needs a 128-thread threadgroup
                // (pipeline-cap gated like the split kernels); other head sizes keep the
                // single-simdgroup flash.
                let flash2_kern = match hd {
                    64 => Some("attnflash2_f16kv_hd64"),
                    128 => Some("attnflash2_f16kv_hd128"),
                    _ => None,
                };
                let flash2 = flash
                    && flash2_kern.is_some_and(|kn| {
                        self.pipelines
                            .get(kn)
                            .map(|pl| pl.max_total_threads_per_threadgroup() >= 128)
                            .unwrap_or(false)
                    });
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
                // The vector flash kernel (llama.cpp flash_attn_ext_vec structure) covers the
                // long-context decode shape split32 serves, at 32 KV positions per simdgroup
                // step instead of 1 — same head-size instantiations and cap gating as flash2.
                let vec_kern = match hd {
                    64 => Some("attnvec_f16kv_hd64"),
                    128 => Some("attnvec_f16kv_hd128"),
                    _ => None,
                };
                let vec = !flash
                    && f16
                    && split
                    && kv_len >= 128
                    && vec_kern.is_some_and(|kn| {
                        self.pipelines
                            .get(kn)
                            .map(|pl| pl.max_total_threads_per_threadgroup() >= 1024)
                            .unwrap_or(false)
                    });
                let split32 = split
                    && !vec
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
                    && (vec
                        || split32
                        || fits(
                            if f16 {
                                "attnsplit_f16kv"
                            } else {
                                "attnsplit_f32"
                            },
                            256,
                        )?);
                let kern = match (flash, f16, split, split32) {
                    (true, ..) if flash2 => flash2_kern.unwrap(),
                    (true, ..) => "attnflash_f16kv",
                    (_, true, true, _) if vec => vec_kern.unwrap(),
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
                    // flash2 runs 4 cooperating simdgroups per (query tile, head) threadgroup
                    let tgw = if flash2 { 128 } else { 32 };
                    self.encode_tg(
                        r,
                        &pso,
                        &[qh.as_ref(), &kbuf.raw, &vbuf.raw, bd.as_ref()],
                        &p,
                        rows.div_ceil(8) * nh * tgw,
                        tgw,
                    );
                } else {
                    // One simdgroup per (query, head); split/vec kernels use NSG simdgroups
                    // per pair, grid still exactly rows*n_head threadgroups.
                    let nsg = if vec || split32 {
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
                // Rows inferred from the bound activation shape (the seam's batched prefill
                // passes [rows, ne]); each row routes independently.
                let rows = g.desc(x).numel() / ne;
                // Device path for K-quant experts (the shapes real MoE checkpoints ship): router
                // GEMV -> on-device top-k -> expert-BATCHED GEMV stages picking their weight
                // slices from the device expert table, weighted sum via a final reduce. Seven
                // dispatches, no expert bytes or activations on the host. Falls back to the host
                // path below for other dtypes (INFR_METAL_NOMOE forces it).
                let moe_kern = |dt: DType| -> Option<(&'static str, &'static str)> {
                    match dt {
                        DType::Q4K => Some(("linear_q4k_moe", "linear_q4k_moe_acc")),
                        DType::Q6K => Some(("linear_q6k_moe", "linear_q6k_moe_acc")),
                        _ => None,
                    }
                };
                let (gdt2, udt2, ddt2) = (
                    g.desc(gate_exps).dtype,
                    g.desc(up_exps).dtype,
                    g.desc(down_exps).dtype,
                );
                let device_ok = n_used <= 16
                    && ne % 256 == 0
                    && nffx % 256 == 0
                    && std::env::var("INFR_METAL_NOMOE").is_err()
                    && moe_kern(gdt2).is_some()
                    && moe_kern(udt2).is_some()
                    && moe_kern(ddt2).is_some();
                if device_ok {
                    let bx = self.ensure_device(r, x);
                    let bd = self.dev_dst(r, dst, rows * ne);
                    // Router logits for ALL rows (one GEMV-per-(row, expert) dispatch over the
                    // dequant-cached router weight), then per-row top-k into the [rows, 32]
                    // expert table (idx in slots 0..16, f32 weights in 16..32).
                    let rw = self.weight_buf(router, g, bindings);
                    let logits = self.scratch_buf(rows * n_expert, 0);
                    let pso = self.pipelines.get("linear_f32")?;
                    let mut p = (rows as u32).to_ne_bytes().to_vec();
                    p.extend_from_slice(&(ne as u32).to_ne_bytes());
                    p.extend_from_slice(&(n_expert as u32).to_ne_bytes());
                    self.encode_tg(
                        r,
                        &pso,
                        &[bx.as_ref(), rw.as_ref(), logits.as_ref()],
                        &p,
                        rows * n_expert * 32,
                        32,
                    );
                    let tbl = self.scratch_buf(rows * 32, 1);
                    let pso = self.pipelines.get("moe_topk")?;
                    let mut p = (n_expert as u32).to_ne_bytes().to_vec();
                    p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                    p.extend_from_slice(&scale.to_ne_bytes());
                    self.encode_tg(r, &pso, &[logits.as_ref(), tbl.as_ref()], &p, rows * 32, 32);
                    // Expert FFN, batched over (chunk rows x selected experts) — one dispatch per
                    // stage per chunk; chunking bounds the expert scratch (a full 8k-row prefill
                    // would need ~0.5 GB of it).
                    let gq = self.weight_qui(gate_exps, g, bindings);
                    let uq = self.weight_qui(up_exps, g, bindings);
                    let dq = self.weight_qui(down_exps, g, bindings);
                    const CHUNK: usize = 256;
                    let chunk = rows.min(CHUNK);
                    let gate_t = self.scratch_buf(chunk * n_used * nffx, 2);
                    let up_t = self.scratch_buf(chunk * n_used * nffx, 3);
                    let act_t = self.scratch_buf(chunk * n_used * nffx, 4);
                    let ydown = self.scratch_buf(chunk * n_used * ne, 5);
                    let act_code: u32 = match act {
                        infr_core::graph::Activation::Silu => 0,
                        infr_core::graph::Activation::Gelu => 1,
                        infr_core::graph::Activation::Sigmoid => 2,
                    };
                    let pack = |in_f: usize, out_f: usize, row0: usize| -> Vec<u8> {
                        let mut p = 1u32.to_ne_bytes().to_vec();
                        p.extend_from_slice(&(in_f as u32).to_ne_bytes());
                        p.extend_from_slice(&(out_f as u32).to_ne_bytes());
                        p.extend_from_slice(&0u32.to_ne_bytes()); // dshift (native kernels)
                        p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                        p.extend_from_slice(&(row0 as u32).to_ne_bytes());
                        p
                    };
                    // Expert-grouped GEMM for prefill-width rows (the llama.cpp mul_mm_id
                    // shape): group the chunk's (token, slot) pairs by expert on-device, then
                    // run the cooperative-GEMM tile per expert over its token group — MMA
                    // compute instead of one GEMV per pair. Small row counts (decode) keep the
                    // GEMV stages below.
                    let grouped =
                        rows >= 16 && ne % 64 == 0 && nffx % 64 == 0 && n_expert <= 1024 && {
                            let kn = if gdt2 == DType::Q4K {
                                "linear_q4k_cmm_id"
                            } else {
                                "linear_q6k_cmm_id"
                            };
                            self.pipelines
                                .get(kn)
                                .map(|pl| pl.max_total_threads_per_threadgroup() >= 128)
                                .unwrap_or(false)
                        };
                    let ids = self.scratch_buf(n_expert * rows.min(CHUNK), 6);
                    let tpe = self.scratch_buf(n_expert, 7);
                    let cmm_id = |dt: DType, scale: bool| -> &'static str {
                        match (dt, scale) {
                            (DType::Q4K, false) => "linear_q4k_cmm_id",
                            (DType::Q4K, true) => "linear_q4k_cmm_id_w",
                            (DType::Q6K, false) => "linear_q6k_cmm_id",
                            _ => "linear_q6k_cmm_id_w",
                        }
                    };
                    for row0 in (0..rows).step_by(CHUNK) {
                        let cr = (rows - row0).min(CHUNK);
                        if grouped {
                            let cap = rows.min(CHUNK);
                            // map: one thread per expert scans the chunk's routing table
                            let pso = self.pipelines.get("moe_map")?;
                            let mut p = (n_expert as u32).to_ne_bytes().to_vec();
                            p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                            p.extend_from_slice(&(cr as u32).to_ne_bytes());
                            p.extend_from_slice(&(row0 as u32).to_ne_bytes());
                            p.extend_from_slice(&(cap as u32).to_ne_bytes());
                            self.encode_tg(
                                r,
                                &pso,
                                &[tbl.as_ref(), ids.as_ref(), tpe.as_ref()],
                                &p,
                                n_expert,
                                n_expert,
                            );
                            let ntt = cr.div_ceil(32);
                            let idpack = |in_f: usize, out_f: usize| -> Vec<u8> {
                                let mut p = (in_f as u32).to_ne_bytes().to_vec();
                                p.extend_from_slice(&(out_f as u32).to_ne_bytes());
                                p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                                p.extend_from_slice(&(cap as u32).to_ne_bytes());
                                p.extend_from_slice(&(row0 as u32).to_ne_bytes());
                                p.extend_from_slice(&(ntt as u32).to_ne_bytes());
                                p
                            };
                            let gemm = |me: &Self,
                                        r: &mut Resident,
                                        kern: &'static str,
                                        xb: &MtlBuffer,
                                        q: &QuiWeight,
                                        db: &MtlBuffer,
                                        in_f: usize,
                                        out_f: usize|
                             -> Result<()> {
                                let pso = me.pipelines.get(kern)?;
                                me.encode_tg(
                                    r,
                                    &pso,
                                    &[
                                        xb,
                                        &q.codes,
                                        &q.scm,
                                        &q.dd,
                                        db,
                                        ids.as_ref(),
                                        tpe.as_ref(),
                                        tbl.as_ref(),
                                    ],
                                    &idpack(in_f, out_f),
                                    n_expert * ntt * (out_f / 64) * 128,
                                    128,
                                );
                                Ok(())
                            };
                            gemm(
                                self,
                                r,
                                cmm_id(gdt2, false),
                                bx.as_ref(),
                                &gq,
                                gate_t.as_ref(),
                                ne,
                                nffx,
                            )?;
                            gemm(
                                self,
                                r,
                                cmm_id(udt2, false),
                                bx.as_ref(),
                                &uq,
                                up_t.as_ref(),
                                ne,
                                nffx,
                            )?;
                            let pso = self.pipelines.get("gatedact_f32")?;
                            let mut p = ((cr * n_used) as u32).to_ne_bytes().to_vec();
                            p.extend_from_slice(&(nffx as u32).to_ne_bytes());
                            p.extend_from_slice(&act_code.to_ne_bytes());
                            p.extend_from_slice(&0u32.to_ne_bytes());
                            self.encode(
                                r,
                                &pso,
                                &[gate_t.as_ref(), up_t.as_ref(), act_t.as_ref()],
                                &p,
                                cr * n_used * nffx,
                            );
                            gemm(
                                self,
                                r,
                                cmm_id(ddt2, true),
                                act_t.as_ref(),
                                &dq,
                                ydown.as_ref(),
                                nffx,
                                ne,
                            )?;
                            let pso = self.pipelines.get("moe_reduce")?;
                            let mut p = (ne as u32).to_ne_bytes().to_vec();
                            p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                            p.extend_from_slice(&(cr as u32).to_ne_bytes());
                            p.extend_from_slice(&(row0 as u32).to_ne_bytes());
                            self.encode(r, &pso, &[ydown.as_ref(), bd.as_ref()], &p, cr * ne);
                            continue;
                        }
                        let pso = self.pipelines.get(moe_kern(gdt2).unwrap().0)?;
                        self.encode_tg(
                            r,
                            &pso,
                            &[
                                bx.as_ref(),
                                &gq.codes,
                                &gq.scm,
                                &gq.dd,
                                gate_t.as_ref(),
                                tbl.as_ref(),
                            ],
                            &pack(ne, nffx, row0),
                            cr * n_used * (nffx / 2) * 32,
                            32,
                        );
                        let pso = self.pipelines.get(moe_kern(udt2).unwrap().0)?;
                        self.encode_tg(
                            r,
                            &pso,
                            &[
                                bx.as_ref(),
                                &uq.codes,
                                &uq.scm,
                                &uq.dd,
                                up_t.as_ref(),
                                tbl.as_ref(),
                            ],
                            &pack(ne, nffx, row0),
                            cr * n_used * (nffx / 2) * 32,
                            32,
                        );
                        let pso = self.pipelines.get("gatedact_f32")?;
                        let mut p = ((cr * n_used) as u32).to_ne_bytes().to_vec();
                        p.extend_from_slice(&(nffx as u32).to_ne_bytes());
                        p.extend_from_slice(&act_code.to_ne_bytes());
                        p.extend_from_slice(&0u32.to_ne_bytes()); // up_off
                        self.encode(
                            r,
                            &pso,
                            &[gate_t.as_ref(), up_t.as_ref(), act_t.as_ref()],
                            &p,
                            cr * n_used * nffx,
                        );
                        let pso = self.pipelines.get(moe_kern(ddt2).unwrap().1)?;
                        self.encode_tg(
                            r,
                            &pso,
                            &[
                                act_t.as_ref(),
                                &dq.codes,
                                &dq.scm,
                                &dq.dd,
                                ydown.as_ref(),
                                tbl.as_ref(),
                            ],
                            &pack(nffx, ne, row0),
                            cr * n_used * (ne / 2) * 32,
                            32,
                        );
                        let pso = self.pipelines.get("moe_reduce")?;
                        let mut p = (ne as u32).to_ne_bytes().to_vec();
                        p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                        p.extend_from_slice(&(cr as u32).to_ne_bytes());
                        p.extend_from_slice(&(row0 as u32).to_ne_bytes());
                        self.encode(r, &pso, &[ydown.as_ref(), bd.as_ref()], &p, cr * ne);
                    }
                    r.loc[dst.0 as usize] = Loc::Device;
                    return Ok(());
                }
                self.ensure_host(r, g, x);
                let xs = r.vals[x.0 as usize].clone();
                // `x` may hold several rows (the seam's batched prefill) — route + run per row,
                // mirroring the CPU reference's row loop.
                let rows = xs.len() / ne;
                // Router (host, mirroring the CPU reference). Structurally the same top-k selection,
                // but the logits use a naive f32 sum here vs the CPU's 8-accumulator dot, so the
                // summation order differs — top-k can pick differently on a near-tie logit.
                let rw = self.weight_host(router, g, bindings);
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
                let mut out = vec![0f32; rows * ne];
                for (row, orow) in out.chunks_mut(ne).enumerate() {
                    let xr = &xs[row * ne..row * ne + ne];
                    let logits: Vec<f32> = (0..n_expert)
                        .map(|e| (0..ne).map(|i| rw[e * ne + i] * xr[i]).sum::<f32>())
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
                    for &e in &idx {
                        // Dequant only this expert's slices, then matvec on the GPU.
                        let gw = bytes_to_f32(&Self::read_bytes_range(gbuf, e * gst, gst), gdt);
                        let uw = bytes_to_f32(&Self::read_bytes_range(ubuf, e * ust, ust), udt);
                        let dw = bytes_to_f32(&Self::read_bytes_range(dbuf, e * dsz, dsz), ddt);
                        let gate = self.gpu_matvec(xr, &gw, ne, nffx)?;
                        let up = self.gpu_matvec(xr, &uw, ne, nffx)?;
                        let actv: Vec<f32> =
                            (0..nffx).map(|i| act_fn(act, gate[i]) * up[i]).collect();
                        let y = self.gpu_matvec(&actv, &dw, nffx, ne)?;
                        let w_e = probs[e] / wsum * scale;
                        for i in 0..ne {
                            orow[i] += w_e * y[i];
                        }
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
                // Device path: each thread owns one channel and its state column — no cross-
                // thread deps, rows loop in-kernel, state updated in the BOUND buffer (the
                // write-back self-copy guard keeps it authoritative). kk > 8 exceeds the
                // kernel's register window; no real conv ships that (qwen3-next uses 4).
                if kk <= 8 && std::env::var("INFR_METAL_NODELTA").is_err() {
                    if let Some(sb) = bindings.get(state) {
                        let bx = self.ensure_device(r, x);
                        let bw = self.weight_buf(weight, g, bindings);
                        let bd = self.dev_dst(r, dst, rr * cc);
                        let sbuf = metal_buf(sb);
                        let i = state.0 as usize;
                        r.dev[i] = Some(Arc::new(sbuf.raw.clone()));
                        r.loc[i] = Loc::Device;
                        let pso = self.pipelines.get("conv1d_silu_f32")?;
                        let mut p = (rr as u32).to_ne_bytes().to_vec();
                        p.extend_from_slice(&(cc as u32).to_ne_bytes());
                        p.extend_from_slice(&(kk as u32).to_ne_bytes());
                        self.encode(
                            r,
                            &pso,
                            &[bx.as_ref(), bw.as_ref(), &sbuf.raw, bd.as_ref()],
                            &p,
                            cc,
                        );
                        r.loc[dst.0 as usize] = Loc::Device;
                        return Ok(());
                    }
                }
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
                // Device path: one threadgroup per value head, one lane per value dim — each
                // lane owns state column S[:, d] (the delta-rule update touches only that
                // column, so the row scan needs no state synchronization). State updates the
                // BOUND buffer in place. Gates match the kernel's shared/thread budgets.
                if kd <= 256
                    && vd.is_multiple_of(32)
                    && vd <= 1024
                    && std::env::var("INFR_METAL_NODELTA").is_err()
                {
                    if let Some(sb) = bindings.get(state) {
                        let fits = self
                            .pipelines
                            .get("deltanet_f32")
                            .map(|pl| pl.max_total_threads_per_threadgroup() >= vd as u64)
                            .unwrap_or(false);
                        if fits {
                            let bq = self.ensure_device(r, q);
                            let bk = self.ensure_device(r, k);
                            let bv = self.ensure_device(r, v);
                            let bb = self.ensure_device(r, b);
                            let ba = self.ensure_device(r, a);
                            let bac = self.weight_buf(a_coef, g, bindings);
                            let bdt = self.weight_buf(dt_bias, g, bindings);
                            let bd = self.dev_dst(r, dst, rr * nv * vd);
                            let sbuf = metal_buf(sb);
                            let i = state.0 as usize;
                            r.dev[i] = Some(Arc::new(sbuf.raw.clone()));
                            r.loc[i] = Loc::Device;
                            let pso = self.pipelines.get("deltanet_f32")?;
                            let mut p = (rr as u32).to_ne_bytes().to_vec();
                            p.extend_from_slice(&(nv as u32).to_ne_bytes());
                            p.extend_from_slice(&(nk as u32).to_ne_bytes());
                            p.extend_from_slice(&(kd as u32).to_ne_bytes());
                            p.extend_from_slice(&(vd as u32).to_ne_bytes());
                            p.extend_from_slice(&eps.to_ne_bytes());
                            self.encode_tg(
                                r,
                                &pso,
                                &[
                                    bq.as_ref(),
                                    bk.as_ref(),
                                    bv.as_ref(),
                                    bb.as_ref(),
                                    ba.as_ref(),
                                    bac.as_ref(),
                                    bdt.as_ref(),
                                    &sbuf.raw,
                                    bd.as_ref(),
                                ],
                                &p,
                                nv * vd,
                                vd,
                            );
                            r.loc[dst.0 as usize] = Loc::Device;
                            return Ok(());
                        }
                    }
                }
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
            // On-device: the fused-QKV prefill emits one of these per projection right after the
            // wide GEMM — a host copy would round-trip the [m, qkv] activation mid-forward.
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
                let bs = self.ensure_device(r, src);
                // The strided write may cover only part of dst — bring the rest along.
                let bd = self.ensure_device(r, dst);
                let mut p = src_off.to_ne_bytes().to_vec();
                p.extend_from_slice(&src_stride.to_ne_bytes());
                p.extend_from_slice(&dst_off.to_ne_bytes());
                p.extend_from_slice(&dst_stride.to_ne_bytes());
                p.extend_from_slice(&rows.to_ne_bytes());
                p.extend_from_slice(&n.to_ne_bytes());
                let pso = self.pipelines.get("copy_strided_f32")?;
                self.encode(
                    r,
                    &pso,
                    &[bs.as_ref(), bd.as_ref()],
                    &p,
                    rows as usize * n as usize,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
        }
        Ok(())
    }
}
