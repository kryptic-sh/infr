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
            Ok(f16s
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect())
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
    let grid_x = total_threads.div_ceil(block_size);
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
                self.dev[i] = Some(crate::RocmBuffer {
                    ptr: p,
                    len: b.len,
                    owned: false,
                });
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
        let raw = read_bytes(b, self.stream);
        let f32s = bytes_to_f32(&raw, dt)?;
        let f16_bytes = f32_to_f16_bytes(&f32s);
        let dq = self.f16_dev(&f16_bytes);
        let ptr = dq.ptr;
        let len = dq.len;
        {
            let mut cache = self.weight_cache.lock().unwrap();
            // Cache owns the device memory (owned: true)
            cache.insert(
                key,
                crate::RocmBuffer {
                    ptr: dq.ptr,
                    len: dq.len,
                    owned: true,
                },
            );
        }
        // Store a non-owned reference in dev so ctx.drop doesn't free it.
        // Prevent dq from dropping (cache owns the allocation now).
        std::mem::forget(dq);
        self.dev[i] = Some(crate::RocmBuffer {
            ptr,
            len,
            owned: false,
        });
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
        run_op(op, g, bindings, pipelines, &mut ctx)?;
        // Sync + check for async errors after each op during bringup.
        let rc = unsafe { ffi::hipStreamSynchronize(stream) };
        if rc != HIP_SUCCESS {
            return Err(be(format!(
                "sync after op {idx} ({kind}): rc={rc}",
                idx = idx,
                kind = op.kind()
            )));
        }
    }

    // Sync after all ops (the final checked sync below re-syncs after writeback).
    unsafe { ffi::hipStreamSynchronize(stream) };
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
            x_stride,
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
            // Per-row stride in elements (0 = packed n_head*head_dim). Mirrors the fused
            // qk_norm_rope stride convention: heads stay packed within a strided row.
            let stride_elems = if x_stride > 0 {
                x_stride as usize
            } else {
                n_head as usize * head_dim as usize
            };
            if dst == x {
                let rope_args = args![
                    arg_ptr(bx_ptr),
                    arg_ptr(bp_ptr),
                    arg_ptr(ff_ptr),
                    arg_i32(rows as i32),
                    arg_i32(n_head as i32),
                    arg_i32(head_dim as i32),
                    arg_i32(rope_dim as i32),
                    arg_f32(theta),
                    arg_i32(x_stride as i32),
                ];
                dispatch_1d(pipelines, ctx.stream, "rope", rows, 256, rope_args)?;
            } else {
                // Copy the FULL (possibly strided) source so both the pass-through dims and the
                // inter-row gaps survive, then rotate in place. A packed input (x_stride == 0)
                // allocs the natural rows*n_head*head_dim; a strided view needs rows*stride so the
                // kernel's off = row*stride + h*head_dim stays in bounds for every row.
                let dd = ctx.zero_dev(rows as usize * stride_elems);
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
                    arg_i32(x_stride as i32),
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
            // Output is ALWAYS a fresh PACKED [rows, n_head, head_dim] buffer: the kernel reads the
            // (possibly strided/interleaved q+g) input and writes the packed query — so no in-place
            // rotation and no strided-source copy (the old copy grabbed a packed prefix of a wider
            // row and then indexed it with the strided stride → out-of-bounds on multi-row prefill).
            // Matches infr-cpu QkNormRope, which always produces a fresh packed `out`.
            let dd = ctx.zero_dev(rows as usize * n_head as usize * head_dim as usize);
            let qnr_args = args![
                arg_ptr(bx_ptr),
                arg_ptr(wptr),
                arg_ptr(bp_ptr),
                arg_ptr(ff_ptr),
                arg_ptr(dd.ptr),
                arg_i32(rows as i32),
                arg_i32(n_head as i32),
                arg_i32(head_dim as i32),
                arg_i32(rope_dim as i32),
                arg_f32(eps),
                arg_f32(theta),
                arg_i32(x_stride as i32),
            ];
            dispatch_1d(pipelines, ctx.stream, "qk_norm_rope", total, 256, qnr_args)?;
            ctx.dev[dst.0 as usize] = Some(dd);
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
                    arg_i32(0), // src_stride (0 = packed = row_stride)
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
            scale,
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
                    arg_f32(scale),
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
            weight_before,
            fused_gate_up,
            ep_band: _ep,
        } => {
            // Router weight [n_expert, ne] (dequantized to f16 and cached — the SAME handle
            // fed to the GEMV below; the previous code discarded it and softmaxed the raw
            // router_x row, selecting bogus "expert" indices out past the expert banks).
            let rw = ctx.dequant_weight_or_cache(router, g, bindings)?;
            let gw_ptr = ctx.dequant_weight_or_cache(gate_exps, g, bindings)?;
            let uw_ptr = if fused_gate_up {
                gw_ptr
            } else {
                ctx.dequant_weight_or_cache(up_exps, g, bindings)?
            };
            let dw_ptr = ctx.dequant_weight_or_cache(down_exps, g, bindings)?;

            let neu = ne as usize;
            let nexp = n_expert as usize;
            let nu = n_used as usize;
            let nfu = n_ff_exp as usize;

            // `x` (and `router_x`, usually the same handle) carry `rows` token rows of `ne`.
            let x_ptr = ctx.ensure_device(x, g, bindings)?;
            let rx_ptr = if router_x != x {
                ctx.ensure_device(router_x, g, bindings)?
            } else {
                x_ptr
            };
            let rows = g.desc(x).numel() / neu;

            // Per-expert down-projection output scale (diffusion-gemma); 1.0 = none.
            let dsc_vals: Vec<f32> = match down_scale {
                Some(sid) => ctx.host_vals(sid, g, bindings)?.to_vec(),
                None => vec![1.0f32; nexp],
            };

            // Router logits = router · router_x, one dot per expert: reuse the linear_f16
            // GEMV to produce [rows, n_expert], then read them back for host-side gating.
            let logits_dev = ctx.zero_dev(rows * nexp);
            dispatch_1d(
                pipelines,
                ctx.stream,
                "linear_f16",
                (rows as u32) * 256,
                256,
                args![
                    arg_ptr(rx_ptr),
                    arg_ptr(rw),
                    arg_ptr(logits_dev.ptr),
                    arg_i32(rows as i32),
                    arg_i32(ne as i32),
                    arg_i32(n_expert as i32),
                ],
            )?;
            unsafe {
                ffi::hipStreamSynchronize(ctx.stream);
            }
            let logits_all: Vec<f32> = {
                let raw = read_bytes(&logits_dev, ctx.stream);
                bytemuck::cast_slice::<u8, f32>(&raw).to_vec()
            };

            let at: i32 = match act {
                infr_core::graph::Activation::Silu => 0,
                infr_core::graph::Activation::Gelu => 1,
                infr_core::graph::Activation::Sigmoid => 2,
            };
            let wb_flag: i32 = if weight_before { 1 } else { 0 };
            // Per-expert byte strides in the (f16) expert banks. Fused gate/up packs BOTH
            // roles per expert as [2*n_ff_exp, ne] (gate rows first, up second), so its expert
            // stride is DOUBLE the split-tensor stride.
            let ge_stride = if fused_gate_up {
                2 * nfu * neu
            } else {
                nfu * neu
            };

            let dd = ctx.zero_dev(rows * neu);
            for row in 0..rows {
                let logits = &logits_all[row * nexp..row * nexp + nexp];
                // Gating: softmax over experts (qwen3moe/…) or per-expert sigmoid (llama4).
                let probs: Vec<f32> = match gating {
                    infr_core::graph::MoeGating::Softmax => {
                        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
                        let sum: f32 = exps.iter().sum();
                        exps.iter().map(|v| v / sum).collect()
                    }
                    infr_core::graph::MoeGating::Sigmoid => {
                        logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
                    }
                };
                let mut idx: Vec<usize> = (0..nexp).collect();
                idx.sort_unstable_by(|&a, &b| {
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                idx.truncate(nu);
                // `norm_w`: renormalize the selected weights to sum to 1 before scaling
                // (softmax MoE); llama4 uses the raw sigmoid prob × scale (no renorm).
                let wsum: f32 = if norm_w {
                    idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20)
                } else {
                    1.0
                };
                let x_row = unsafe { (x_ptr as *mut u8).add(row * neu * 4) as *mut c_void };
                let dst_row = unsafe { (dd.ptr as *mut u8).add(row * neu * 4) as *mut c_void };
                for &ei in &idx {
                    let w = probs[ei] / wsum * scale;
                    let gs = unsafe { (gw_ptr as *mut u8).add(ei * ge_stride * 2) as *mut c_void };
                    let us = if fused_gate_up {
                        unsafe {
                            (gw_ptr as *mut u8).add((ei * ge_stride + nfu * neu) * 2) as *mut c_void
                        }
                    } else {
                        unsafe { (uw_ptr as *mut u8).add(ei * nfu * neu * 2) as *mut c_void }
                    };
                    let ds = unsafe { (dw_ptr as *mut u8).add(ei * neu * nfu * 2) as *mut c_void };
                    let dsc = dsc_vals.get(ei).copied().unwrap_or(1.0);
                    dispatch_1d(
                        pipelines,
                        ctx.stream,
                        "moe_ffn_expert",
                        n_ff_exp,
                        256,
                        args![
                            arg_ptr(x_row),
                            arg_ptr(gs),
                            arg_ptr(us),
                            arg_ptr(ds),
                            arg_ptr(dst_row),
                            arg_i32(ne as i32),
                            arg_i32(n_ff_exp as i32),
                            arg_i32(at),
                            arg_f32(w),
                            arg_f32(dsc),
                            arg_i32(wb_flag),
                        ],
                    )?;
                }
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
            // Host-side state update: the returned state is the trailing `km1` columns of the
            // virtual sequence seq = [state ‖ x] (km1 warmup columns then `rows` input columns),
            // i.e. new_state[j] = seq[rows + j] for j in 0..km1. This chains correctly for any
            // `rows`: for `rows >= km1` all km1 columns come from the last km1 input rows; for
            // `rows < km1` the leading entries carry over from the old state tail. For `rows == 1`
            // it reduces to the old "drop oldest, append x[0]" shift (decode is bit-identical).
            let km1 = (kernel - 1) as usize;
            let ch = channels as usize;
            let rows_u = rows as usize;
            let hs = ctx.host_vals(state, g, bindings)?.to_vec();
            let hx = ctx.host_vals(x, g, bindings)?.to_vec();
            let mut ns = vec![0f32; km1 * ch];
            for j in 0..km1 {
                let idx = rows_u + j; // virtual-sequence index of new_state column j
                for c in 0..ch {
                    ns[j * ch + c] = if idx < km1 {
                        hs[idx * ch + c] // still inside the old state tail
                    } else {
                        hx[(idx - km1) * ch + c] // an input column
                    };
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
            eps,
            src_stride,
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
                    arg_f32(eps),
                    arg_i32(src_stride as i32),
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
