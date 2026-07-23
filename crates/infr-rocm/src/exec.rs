//! Graph execution: walk ops → resolve bound buffers → dispatch HIP kernels.
//!
//! Quantized weight tensors are dequantized to f16 on the host on first touch and
//! cached by the raw device-pointer address of their bound buffer.

use crate::ffi::{self, HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE, HIP_SUCCESS};
use crate::kernels::Pipelines;
use half::f16;
use infr_core::backend::{Bindings, GraphPlan, Plan};
use infr_core::error::{Error, Result};
use infr_core::graph::{AttnMask, Graph, Op, TensorKind};
use infr_core::tensor::{DType, TensorId};
use infr_gguf::dequant;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

fn be(msg: impl std::fmt::Display) -> Error {
    Error::backend(msg)
}

fn rocm_buf(b: &dyn infr_core::backend::Buffer) -> &crate::RocmBuffer {
    b.as_any()
        .downcast_ref::<crate::RocmBuffer>()
        .expect("rocm backend: buffer is not a RocmBuffer")
}

fn read_bytes(b: &crate::RocmBuffer, stream: ffi::hipStream_t) -> Vec<u8> {
    let mut v = vec![0u8; b.len];
    if b.len > 0 {
        unsafe {
            ffi::hipMemcpy(
                v.as_mut_ptr() as *mut c_void,
                b.ptr,
                b.len,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
        }
        unsafe {
            ffi::hipStreamSynchronize(stream);
        }
    }
    v
}

fn bytes_to_f32(bytes: &[u8], dtype: DType) -> Result<Vec<f32>> {
    match dtype {
        DType::F32 => {
            // Raw f32 bytes — reinterpret directly.
            let f32s: &[f32] = bytemuck::cast_slice(bytes);
            Ok(f32s.to_vec())
        }
        DType::F16 => {
            // Raw f16 bytes — convert each half to f32.
            let f16s: &[u16] = bytemuck::cast_slice(bytes);
            Ok(f16s.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect())
        }
        DType::I32 => {
            // Bias / position tensor — bitcast i32 to f32.
            let i32s: &[i32] = bytemuck::cast_slice(bytes);
            Ok(i32s.iter().map(|&v| f32::from_bits(v as u32)).collect())
        }
        _ => dequant::dequant_block(dtype, bytes)
            .map_err(|e| be(format!("dequant {dtype:?} weight: {e}"))),
    }
}

fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        let h = f16::from_f32(*x);
        out.extend_from_slice(&h.to_bits().to_le_bytes());
    }
    out
}

// ── Kernel dispatch helpers ──────────────────────────────────────────────────

fn dispatch_1d(
    pipelines: &Pipelines,
    stream: ffi::hipStream_t,
    kernel_name: &'static str,
    total_threads: u32,
    block_size: u32,
    args: Vec<Vec<u8>>,
) -> Result<()> {
    let func = pipelines.get(kernel_name)?;
    let grid_x = (total_threads + block_size - 1) / block_size;
    let mut storage = args;
    let mut arg_ptrs: Vec<*mut c_void> = Vec::with_capacity(storage.len());
    for ab in storage.iter_mut() {
        arg_ptrs.push(ab.as_mut_ptr() as *mut c_void);
    }
    let rc = unsafe {
        ffi::hipModuleLaunchKernel(
            func,
            grid_x,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            arg_ptrs.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if rc != HIP_SUCCESS {
        return Err(be(format!("hipModuleLaunchKernel({kernel_name}): rc={rc}")));
    }
    Ok(())
}

fn arg_ptr(p: *mut c_void) -> Vec<u8> {
    (p as u64).to_le_bytes().to_vec()
}
fn arg_i32(v: i32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn arg_u32(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn arg_f32(v: f32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

// ── ExecCtx ──────────────────────────────────────────────────────────────────

struct ExecCtx<'a> {
    dev: Vec<Option<crate::RocmBuffer>>,
    vals: Vec<Option<Vec<f32>>>,
    weight_cache: &'a Mutex<HashMap<usize, crate::RocmBuffer>>,
    stream: ffi::hipStream_t,
}

impl<'a> ExecCtx<'a> {
    fn f32_dev(&self, data: &[f32]) -> crate::RocmBuffer {
        let bytes = bytemuck::cast_slice::<f32, u8>(data);
        let mut buf = crate::RocmBuffer::alloc(bytes.len().max(1), self.stream);
        buf.upload(bytes, self.stream);
        buf
    }

    fn f16_dev(&self, data: &[u8]) -> crate::RocmBuffer {
        let mut buf = crate::RocmBuffer::alloc(data.len().max(1), self.stream);
        buf.upload(data, self.stream);
        buf
    }

    fn zero_dev(&self, n: usize) -> crate::RocmBuffer {
        // Allocate in BYTES — n is ELEMENTS of f32 (4 bytes each).
        crate::RocmBuffer::alloc_zero((n * 4).max(1), self.stream)
    }

    fn host_vals(&mut self, id: TensorId, g: &Graph, bindings: &Bindings) -> Result<&[f32]> {
        let i = id.0 as usize;
        if self.vals[i].is_none() {
            let decl = &g.tensors[i];
            let val = match decl.kind {
                TensorKind::Input | TensorKind::Weight => {
                    let b = rocm_buf(bindings.get(id).expect("rocm: unbound Input/Weight"));
                    let raw = read_bytes(b, self.stream);
                    bytes_to_f32(&raw, decl.desc.dtype)?
                }
                TensorKind::Internal | TensorKind::Output => {
                    if let Some(ref db) = self.dev[i] {
                        let raw = read_bytes(db, self.stream);
                        bytes_to_f32(&raw, decl.desc.dtype)?
                    } else {
                        vec![0f32; decl.desc.numel()]
                    }
                }
            };
            self.vals[i] = Some(val);
        }
        Ok(self.vals[i].as_ref().unwrap())
    }

    fn ensure_device(
        &mut self,
        id: TensorId,
        g: &Graph,
        bindings: &Bindings,
    ) -> Result<*mut c_void> {
        let i = id.0 as usize;
        if let Some(ref db) = self.dev[i] {
            return Ok(db.ptr);
        }
        // For Input/Weight tensors, use the bound buffer directly (no host download).
        let decl = &g.tensors[i];
        let ptr = match decl.kind {
            TensorKind::Input | TensorKind::Weight => {
                let b = rocm_buf(bindings.get(id).expect("rocm: unbound Input/Weight"));
                let p = b.ptr;
                // Track in dev so subsequent accesses find it.
                self.dev[i] = Some(crate::RocmBuffer { ptr: p, len: b.len, owned: false });
                p
            }
            TensorKind::Internal | TensorKind::Output => {
                // Not yet produced — allocate a zero-filled buffer.
                let db = self.zero_dev(decl.desc.numel());
                let p = db.ptr;
                self.dev[i] = Some(db);
                p
            }
        };
        Ok(ptr)
    }

    fn set_dev(&mut self, id: TensorId, data: &[f32]) {
        self.dev[id.0 as usize] = Some(self.f32_dev(data));
    }

    fn dequant_weight_or_cache(
        &mut self,
        id: TensorId,
        g: &Graph,
        bindings: &Bindings,
    ) -> Result<*mut c_void> {
        let i = id.0 as usize;
        let b = rocm_buf(bindings.get(id).expect("rocm: unbound Weight"));
        let key = b.ptr as usize;
        {
            let cache = self.weight_cache.lock().unwrap();
            if let Some(cached) = cache.get(&key) {
                return Ok(cached.ptr);
            }
        }
        let dt = g.desc(id).dtype;
        eprintln!("[rocm] dequant weight id={i} dtype={dt:?} bytes={nbytes}", i=id.0, dt=dt, nbytes=b.len);
        let raw = read_bytes(b, self.stream);
        let f32s = bytes_to_f32(&raw, dt)?;
        let f16_bytes = f32_to_f16_bytes(&f32s);
        let dq = self.f16_dev(&f16_bytes);
        let ptr = dq.ptr;
        {
            let mut cache = self.weight_cache.lock().unwrap();
            cache.insert(
                key,
                crate::RocmBuffer {
                    ptr: dq.ptr,
                    len: dq.len,
                    owned: false,
                },
            );
        }
        self.dev[i] = Some(dq);
        Ok(ptr)
    }
}

// ── Main execute walk ────────────────────────────────────────────────────────

pub fn execute_graph(
    pipelines: &Pipelines,
    weight_cache: &Mutex<HashMap<usize, crate::RocmBuffer>>,
    stream: ffi::hipStream_t,
    plan: &dyn Plan,
    bindings: &Bindings,
) -> Result<()> {
    let g = &plan
        .as_any()
        .downcast_ref::<GraphPlan>()
        .expect("rocm backend: plan is not a GraphPlan")
        .graph;
    let n = g.tensors.len();
    let mut ctx = ExecCtx {
        dev: (0..n).map(|_| None).collect(),
        vals: (0..n).map(|_| None).collect(),
        weight_cache,
        stream,
    };

    for (idx, op) in g.ops.iter().enumerate() {
        eprintln!("[rocm] op {idx}/{len}: {kind}", idx = idx, len = g.ops.len(), kind = op.kind());
        run_op(op, g, bindings, pipelines, &mut ctx)?;
        // Sync + check for async errors after each op during bringup.
        let rc = unsafe { ffi::hipStreamSynchronize(stream) };
        if rc != HIP_SUCCESS {
            return Err(be(format!("sync after op {idx} ({kind}): rc={rc}", idx=idx, kind=op.kind())));
        }
    }

    // Sync after all ops
    eprintln!("[rocm] syncing stream...");
    let rc = unsafe { ffi::hipStreamSynchronize(stream) };
    eprintln!("[rocm] sync done (rc={rc})");
    let direct = g.in_place_inputs();
    for (i, decl) in g.tensors.iter().enumerate() {
        let id = TensorId(i as u32);
        let wb = matches!(decl.kind, TensorKind::Output)
            || (decl.kind == TensorKind::Input
                && decl.desc.dtype == DType::F32
                && !direct.contains(&id));
        if !wb {
            continue;
        }
        if let Some(b) = bindings.get(id) {
            let dst = rocm_buf(b);
            if let Some(ref dev_buf) = ctx.dev[i] {
                if dev_buf.len > 0 {
                    unsafe {
                        ffi::hipMemcpyDtoD(dst.ptr, dev_buf.ptr, dev_buf.len.min(dst.len));
                    }
                }
            } else if let Some(ref vals) = ctx.vals[i] {
                let bytes = bytemuck::cast_slice::<f32, u8>(vals);
                let n = bytes.len().min(dst.len);
                if n > 0 {
                    unsafe {
                        ffi::hipMemcpy(
                            dst.ptr,
                            bytes.as_ptr() as *const c_void,
                            n,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        );
                    }
                }
            }
        }
    }

    let rc = unsafe { ffi::hipStreamSynchronize(stream) };
    if rc != HIP_SUCCESS {
        return Err(be(format!("hipStreamSynchronize: rc={rc}")));
    }
    Ok(())
}

// ── Per-op dispatch ──────────────────────────────────────────────────────────

macro_rules! args { ($($e:expr),* $(,)?) => { vec![$($e),*] }; }

fn run_op(
    op: &Op,
    g: &Graph,
    bindings: &Bindings,
    pipelines: &Pipelines,
    ctx: &mut ExecCtx,
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
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * dim as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "rmsnorm",
                rows,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(wptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(dim as i32),
                    arg_f32(eps),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::RmsNormAdd {
            x,
            weight,
            dst,
            rows,
            dim,
            eps,
        } => {
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(dst, g, bindings)?;
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            let dd = ctx.dev[dst.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "rmsnorm_add",
                rows,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(wptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(dim as i32),
                    arg_f32(eps),
                ],
            )?;
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
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            let dd = ctx.zero_dev(m as usize * out_f as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            let wptr_off = unsafe { (wptr as *mut u8).add(w_off as usize * 2) as *mut c_void };
            dispatch_1d(
                pipelines,
                ctx.stream,
                "linear_f16",
                m * 256,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(wptr_off),
                    arg_ptr(dd.ptr),
                    arg_i32(m as i32),
                    arg_i32(in_f as i32),
                    arg_i32(out_f as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Softmax {
            x,
            dst,
            rows,
            dim,
            scale,
            scale_buf,
        } => {
            ctx.ensure_device(x, g, bindings)?;
            let s = if let Some(sid) = scale_buf {
                ctx.host_vals(sid, g, bindings)?
                    .first()
                    .copied()
                    .unwrap_or(scale)
            } else {
                scale
            };
            let dd = ctx.zero_dev(rows as usize * dim as usize);
            let bx_ptr = ctx.dev[x.0 as usize].as_ref().unwrap().ptr;
            let dd_ptr = dd.ptr;
            dispatch_1d(
                pipelines,
                ctx.stream,
                "softmax",
                rows,
                256,
                args![
                    arg_ptr(bx_ptr),
                    arg_ptr(dd_ptr),
                    arg_i32(rows as i32),
                    arg_i32(dim as i32),
                    arg_f32(s),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::QkNorm {
            x,
            weight,
            dst,
            rows,
            n_head,
            head_dim,
            eps,
            x_stride,
        } => {
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n_head as usize * head_dim as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "qk_norm",
                rows * n_head,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(wptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n_head as i32),
                    arg_i32(head_dim as i32),
                    arg_f32(eps),
                    arg_i32(x_stride as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::GatedRmsNorm {
            x,
            weight,
            gate,
            dst,
            rows,
            n_head,
            head_dim,
            eps,
        } => {
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(gate, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n_head as usize * head_dim as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            let bg = ctx.dev[gate.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "gated_rmsnorm",
                rows * n_head,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(wptr),
                    arg_ptr(bg.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n_head as i32),
                    arg_i32(head_dim as i32),
                    arg_f32(eps),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
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
            x_stride: _x_stride,
        } => {
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(positions, g, bindings)?;
            let ff_ptr = if let Some(fid) = freq_factors {
                ctx.ensure_device(fid, g, bindings)?;
                ctx.dev[fid.0 as usize].as_ref().unwrap().ptr
            } else {
                std::ptr::null_mut()
            };
            // Re-fetch after ensure_device calls (borrow lifetime)
            let bx_ptr = ctx.dev[x.0 as usize].as_ref().unwrap().ptr;
            let bp_ptr = ctx.dev[positions.0 as usize].as_ref().unwrap().ptr;
            let rope_args = args![
                arg_ptr(bx_ptr),
                arg_ptr(bp_ptr),
                arg_ptr(ff_ptr),
                arg_i32(rows as i32),
                arg_i32(n_head as i32),
                arg_i32(head_dim as i32),
                arg_i32(rope_dim as i32),
                arg_f32(theta),
            ];
            if dst == x {
                dispatch_1d(pipelines, ctx.stream, "rope", rows, 256, rope_args)?;
            } else {
                let dd = ctx.zero_dev(rows as usize * n_head as usize * head_dim as usize);
                unsafe {
                    ffi::hipMemcpyDtoD(
                        dd.ptr,
                        bx_ptr,
                        dd.len.min(ctx.dev[x.0 as usize].as_ref().unwrap().len),
                    );
                }
                let dst_args = args![
                    arg_ptr(dd.ptr),
                    arg_ptr(bp_ptr),
                    arg_ptr(ff_ptr),
                    arg_i32(rows as i32),
                    arg_i32(n_head as i32),
                    arg_i32(head_dim as i32),
                    arg_i32(rope_dim as i32),
                    arg_f32(theta),
                ];
                dispatch_1d(pipelines, ctx.stream, "rope", rows, 256, dst_args)?;
                ctx.dev[dst.0 as usize] = Some(dd);
            }
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
            eps,
            theta,
            freq_factors,
            x_stride,
        } => {
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(positions, g, bindings)?;
            let ff_ptr = if let Some(fid) = freq_factors {
                ctx.ensure_device(fid, g, bindings)?;
                ctx.dev[fid.0 as usize].as_ref().unwrap().ptr
            } else {
                std::ptr::null_mut()
            };
            let bx_ptr = ctx.dev[x.0 as usize].as_ref().unwrap().ptr;
            let bp_ptr = ctx.dev[positions.0 as usize].as_ref().unwrap().ptr;
            let total = rows * n_head;
            let qnr_args = args![
                arg_ptr(bx_ptr),
                arg_ptr(wptr),
                arg_ptr(bp_ptr),
                arg_ptr(ff_ptr),
                arg_i32(rows as i32),
                arg_i32(n_head as i32),
                arg_i32(head_dim as i32),
                arg_i32(rope_dim as i32),
                arg_f32(eps),
                arg_f32(theta),
                arg_i32(x_stride as i32),
            ];
            if dst == x {
                dispatch_1d(pipelines, ctx.stream, "qk_norm_rope", total, 256, qnr_args)?;
            } else {
                let dd = ctx.zero_dev(rows as usize * n_head as usize * head_dim as usize);
                unsafe {
                    ffi::hipMemcpyDtoD(
                        dd.ptr,
                        bx_ptr,
                        dd.len.min(ctx.dev[x.0 as usize].as_ref().unwrap().len),
                    );
                }
                let dst_args = args![
                    arg_ptr(dd.ptr),
                    arg_ptr(wptr),
                    arg_ptr(bp_ptr),
                    arg_ptr(ff_ptr),
                    arg_i32(rows as i32),
                    arg_i32(n_head as i32),
                    arg_i32(head_dim as i32),
                    arg_i32(rope_dim as i32),
                    arg_f32(eps),
                    arg_f32(theta),
                    arg_i32(x_stride as i32),
                ];
                dispatch_1d(pipelines, ctx.stream, "qk_norm_rope", total, 256, dst_args)?;
                ctx.dev[dst.0 as usize] = Some(dd);
            }
        }
        Op::WriteKv {
            src,
            cache,
            pos,
            rows,
            row_stride,
        } => {
            ctx.ensure_device(src, g, bindings)?;
            let bs = ctx.dev[src.0 as usize].as_ref().unwrap();
            let bc = rocm_buf(bindings.get(cache).expect("rocm: unbound KV cache"));
            dispatch_1d(
                pipelines,
                ctx.stream,
                "write_kv",
                rows,
                256,
                args![
                    arg_ptr(bs.ptr),
                    arg_ptr(bc.ptr),
                    arg_i32(pos as i32),
                    arg_i32(rows as i32),
                    arg_i32(row_stride as i32),
                ],
            )?;
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
            ctx.ensure_device(q, g, bindings)?;
            let bq = ctx.dev[q.0 as usize].as_ref().unwrap();
            let bk = rocm_buf(bindings.get(k_cache).expect("rocm: unbound K cache"));
            let bv = rocm_buf(bindings.get(v_cache).expect("rocm: unbound V cache"));
            let dd = ctx.zero_dev(rows as usize * n_head as usize * head_dim as usize);
            let (mt, swa): (i32, i32) = match mask {
                AttnMask::Causal => (0, 0),
                AttnMask::SlidingWindow(w) => (1, w as i32),
                AttnMask::Canvas { lo } => (2, lo as i32),
            };
            dispatch_1d(
                pipelines,
                ctx.stream,
                "attention",
                rows * n_head,
                256,
                args![
                    arg_ptr(bq.ptr),
                    arg_ptr(bk.ptr),
                    arg_ptr(bv.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(kv_len as i32),
                    arg_i32(n_head as i32),
                    arg_i32(n_kv as i32),
                    arg_i32(head_dim as i32),
                    arg_f32(scale),
                    arg_i32(pos as i32),
                    arg_i32(mt),
                    arg_i32(swa),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::GatedAct {
            gate,
            up,
            dst,
            rows,
            nff,
            act,
            up_off,
            up_stride,
            gate_stride,
            gate_block_width,
        } => {
            ctx.ensure_device(gate, g, bindings)?;
            ctx.ensure_device(up, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * nff as usize);
            let bg = ctx.dev[gate.0 as usize].as_ref().unwrap();
            let bu = ctx.dev[up.0 as usize].as_ref().unwrap();
            let at: i32 = match act {
                infr_core::graph::Activation::Silu => 0,
                infr_core::graph::Activation::Gelu => 1,
                infr_core::graph::Activation::Sigmoid => 2,
            };
            dispatch_1d(
                pipelines,
                ctx.stream,
                "gated_act",
                rows,
                256,
                args![
                    arg_ptr(bg.ptr),
                    arg_ptr(bu.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(nff as i32),
                    arg_i32(at),
                    arg_i32(up_off as i32),
                    arg_i32(up_stride as i32),
                    arg_i32(gate_stride as i32),
                    arg_i32(gate_block_width as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::GatedActFused {
            gu,
            dst,
            rows,
            nff,
            act,
        } => {
            ctx.ensure_device(gu, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * nff as usize);
            let bgu = ctx.dev[gu.0 as usize].as_ref().unwrap();
            let at: i32 = match act {
                infr_core::graph::Activation::Silu => 0,
                infr_core::graph::Activation::Gelu => 1,
                infr_core::graph::Activation::Sigmoid => 2,
            };
            dispatch_1d(
                pipelines,
                ctx.stream,
                "gated_act",
                rows,
                256,
                args![
                    arg_ptr(bgu.ptr),
                    arg_ptr(bgu.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(nff as i32),
                    arg_i32(at),
                    arg_i32(nff as i32),
                    arg_i32((2 * nff) as i32),
                    arg_i32((2 * nff) as i32),
                    arg_i32(0),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Add { a, b, dst, n } => {
            ctx.ensure_device(a, g, bindings)?;
            ctx.ensure_device(b, g, bindings)?;
            let dd = ctx.zero_dev(n as usize);
            let ba = ctx.dev[a.0 as usize].as_ref().unwrap();
            let bb = ctx.dev[b.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "add",
                n,
                256,
                args![
                    arg_ptr(ba.ptr),
                    arg_ptr(bb.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::AddBias {
            x,
            bias,
            dst,
            rows,
            n,
        } => {
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(bias, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            let bb = ctx.dev[bias.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "add_bias",
                rows,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(bb.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Scale { x, dst, s, n } => {
            ctx.ensure_device(x, g, bindings)?;
            let dd = ctx.zero_dev(n as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "scale",
                n,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(dd.ptr),
                    arg_f32(s),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::MulVec {
            x,
            vec,
            dst,
            rows,
            n,
        } => {
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(vec, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            let bv = ctx.dev[vec.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "mul_vec",
                rows,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(bv.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Softcap { x, dst, cap, n } => {
            ctx.ensure_device(x, g, bindings)?;
            let dd = ctx.zero_dev(n as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "softcap",
                n,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(dd.ptr),
                    arg_f32(cap),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Copy {
            src,
            src_off,
            dst,
            dst_off,
            n,
        } => {
            ctx.ensure_device(src, g, bindings)?;
            let dd = ctx.zero_dev((dst_off + n) as usize);
            let bs = ctx.dev[src.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "copy",
                n,
                256,
                args![
                    arg_ptr(bs.ptr),
                    arg_i32(src_off as i32),
                    arg_ptr(dd.ptr),
                    arg_i32(dst_off as i32),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
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
            ctx.ensure_device(src, g, bindings)?;
            let dd =
                ctx.zero_dev(rows as usize * (dst_off as usize + n as usize + dst_stride as usize));
            let bs = ctx.dev[src.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "copy_strided",
                rows,
                256,
                args![
                    arg_ptr(bs.ptr),
                    arg_i32(src_off as i32),
                    arg_i32(src_stride as i32),
                    arg_ptr(dd.ptr),
                    arg_i32(dst_off as i32),
                    arg_i32(dst_stride as i32),
                    arg_i32(rows as i32),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::EmbedGather {
            ids,
            table,
            dst,
            rows,
            ne,
            scale: _scale,
        } => {
            let wptr = ctx.dequant_weight_or_cache(table, g, bindings)?;
            ctx.ensure_device(ids, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * ne as usize);
            let bid = ctx.dev[ids.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "embed_gather",
                rows,
                256,
                args![
                    arg_ptr(bid.ptr),
                    arg_ptr(wptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(ne as i32),
                    arg_f32(_scale),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Argmax { x, dst, n, rows } => {
            ctx.ensure_device(x, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "argmax",
                rows,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::ArgmaxProb { .. } => return Err(be("ArgmaxProb: Phase 2")),
        Op::Sample { .. } => return Err(be("Sample: Phase 2")),

        Op::MoeFfn {
            x,
            router_x,
            router,
            gate_exps,
            up_exps,
            down_exps,
            down_scale,
            dst,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            scale,
            act,
            gating,
            norm_w,
            weight_before: _wb,
            fused_gate_up,
            ep_band: _ep,
        } => {
            ctx.ensure_device(x, g, bindings)?;
            let _rw = ctx.dequant_weight_or_cache(router, g, bindings)?;
            let gw_ptr = ctx.dequant_weight_or_cache(gate_exps, g, bindings)?;
            let uw_ptr = if fused_gate_up {
                gw_ptr
            } else {
                ctx.dequant_weight_or_cache(up_exps, g, bindings)?
            };
            let dw_ptr = ctx.dequant_weight_or_cache(down_exps, g, bindings)?;

            let router_out = if router_x != x {
                ctx.host_vals(router_x, g, bindings)?.to_vec()
            } else {
                ctx.host_vals(x, g, bindings)?.to_vec()
            };

            let nu = n_used as usize;
            let neu = ne as usize;
            let nfu = n_ff_exp as usize;

            let probs: Vec<f32> = match gating {
                infr_core::graph::MoeGating::Softmax => {
                    let max = router_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = router_out.iter().map(|v| (v - max).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    exps.iter().map(|v| v / sum).collect()
                }
                infr_core::graph::MoeGating::Sigmoid => router_out
                    .iter()
                    .map(|v| 1.0 / (1.0 + (-v).exp()))
                    .collect(),
            };
            let mut idx: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_k: Vec<(usize, f32)> = idx.into_iter().take(nu).collect();
            let sum_w: f32 = if norm_w {
                top_k.iter().map(|(_, w)| w).sum::<f32>().max(1e-9)
            } else {
                1.0
            };

            let dd = ctx.zero_dev(neu);
            let at: i32 = match act {
                infr_core::graph::Activation::Silu => 0,
                infr_core::graph::Activation::Gelu => 1,
                infr_core::graph::Activation::Sigmoid => 2,
            };
            // Pre-fetch down_scale values to avoid borrowing ctx inside the loop
            let dsc_vals: Vec<f32> = match down_scale {
                Some(sid) => ctx.host_vals(sid, g, bindings)?.to_vec(),
                None => vec![1.0f32; n_expert as usize],
            };
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            for &(ei, prob) in &top_k {
                let w = prob / sum_w * scale;
                let eo = ei * nfu * neu;
                let gs = unsafe { (gw_ptr as *mut u8).add(eo * 2) as *mut c_void };
                let us = if fused_gate_up {
                    unsafe { (gw_ptr as *mut u8).add((eo + nfu * neu) * 2) as *mut c_void }
                } else {
                    unsafe { (uw_ptr as *mut u8).add(eo * 2) as *mut c_void }
                };
                let ds = unsafe { (dw_ptr as *mut u8).add(eo * 2) as *mut c_void };
                let dsc = dsc_vals.get(ei).copied().unwrap_or(1.0);
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "moe_ffn_expert",
                    n_ff_exp,
                    256,
                    args![
                        arg_ptr(bx.ptr),
                        arg_ptr(gs),
                        arg_ptr(us),
                        arg_ptr(ds),
                        arg_ptr(dd.ptr),
                        arg_i32(ne as i32),
                        arg_i32(n_ff_exp as i32),
                        arg_i32(at),
                        arg_f32(w),
                        arg_f32(dsc),
                    ],
                )?;
            }
            ctx.dev[dst.0 as usize] = Some(dd);
        }

        Op::Conv1dSilu {
            x,
            weight,
            state,
            dst,
            rows,
            channels,
            kernel,
        } => {
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(state, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * channels as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            let bst = ctx.dev[state.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "conv1d_silu",
                rows,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(wptr),
                    arg_ptr(bst.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(channels as i32),
                    arg_i32(kernel as i32),
                ],
            )?;
            // Host-side state shift
            let km1 = (kernel - 1) as usize;
            let ch = channels as usize;
            let hs = ctx.host_vals(state, g, bindings)?.to_vec();
            let hx = ctx.host_vals(x, g, bindings)?.to_vec();
            let mut ns = hs.clone();
            for r in 0..(rows as usize) {
                for j in 0..km1.saturating_sub(1) {
                    for c in 0..ch {
                        ns[j * ch + c] = hs[(j + 1) * ch + c];
                    }
                }
                if km1 > 0 {
                    let last = km1 - 1;
                    for c in 0..ch {
                        ns[last * ch + c] = hx[r * ch + c];
                    }
                }
            }
            ctx.set_dev(state, &ns);
            ctx.dev[dst.0 as usize] = Some(dd);
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
            n_khead,
            n_vhead,
            head_k,
            head_v,
            ..
        } => {
            ctx.ensure_device(q, g, bindings)?;
            ctx.ensure_device(k, g, bindings)?;
            ctx.ensure_device(v, g, bindings)?;
            ctx.ensure_device(b, g, bindings)?;
            ctx.ensure_device(a, g, bindings)?;
            let ac = ctx.dequant_weight_or_cache(a_coef, g, bindings)?;
            let dt = ctx.dequant_weight_or_cache(dt_bias, g, bindings)?;
            ctx.ensure_device(state, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n_vhead as usize * head_v as usize);
            let bq = ctx.dev[q.0 as usize].as_ref().unwrap();
            let bk = ctx.dev[k.0 as usize].as_ref().unwrap();
            let bv = ctx.dev[v.0 as usize].as_ref().unwrap();
            let bb = ctx.dev[b.0 as usize].as_ref().unwrap();
            let ba = ctx.dev[a.0 as usize].as_ref().unwrap();
            let bst = ctx.dev[state.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "deltanet",
                n_vhead,
                256,
                args![
                    arg_ptr(bq.ptr),
                    arg_ptr(bk.ptr),
                    arg_ptr(bv.ptr),
                    arg_ptr(bb.ptr),
                    arg_ptr(ba.ptr),
                    arg_ptr(ac),
                    arg_ptr(dt),
                    arg_ptr(bst.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n_khead as i32),
                    arg_i32(n_vhead as i32),
                    arg_i32(head_k as i32),
                    arg_i32(head_v as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }

        Op::MoeSharedExpertAdd {
            moe,
            shexp,
            gate,
            dst,
            rows,
            n,
        } => {
            ctx.ensure_device(moe, g, bindings)?;
            ctx.ensure_device(shexp, g, bindings)?;
            ctx.ensure_device(gate, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n as usize);
            let bm = ctx.dev[moe.0 as usize].as_ref().unwrap();
            let bs = ctx.dev[shexp.0 as usize].as_ref().unwrap();
            let bg = ctx.dev[gate.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "moe_shared_expert_add",
                rows,
                256,
                args![
                    arg_ptr(bm.ptr),
                    arg_ptr(bs.ptr),
                    arg_ptr(bg.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
    }
    Ok(())
}
