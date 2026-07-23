//! The ROCm/HIP backend — mirrors `infr-metal`'s structure: backend struct, buffer,
//! and the `Backend` trait impl.
//!
//! Compiled only when `cfg(all(target_os = "linux", feature = "rocm"))`.

use crate::exec;
use crate::ffi::{self, HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE, HIP_SUCCESS};
use crate::kernels::Pipelines;
use infr_core::backend::{
    Backend, Bindings, Buffer, BufferUsage, Capabilities, GraphPlan, Plan, ProgressScope,
};
use infr_core::error::{Error, Result};
use infr_core::graph::Graph;
use std::ffi::{c_int, c_void};
use std::sync::{Arc, Mutex};

fn be(msg: impl std::fmt::Display) -> Error {
    Error::backend(msg)
}

// ── RocmBuffer ───────────────────────────────────────────────────────────────

/// A device buffer allocated with `hipMalloc`.
pub struct RocmBuffer {
    /// Device pointer (null if len == 0).
    pub(crate) ptr: *mut c_void,
    /// Byte length.
    pub(crate) len: usize,
    /// Whether `drop` should call `hipFree` (false for a slice/view into another buffer).
    pub(crate) owned: bool,
}

// Raw device pointers are Send/Sync (they identify a VRAM region, not a CPU address).
unsafe impl Send for RocmBuffer {}
unsafe impl Sync for RocmBuffer {}

impl RocmBuffer {
    /// Allocate `bytes` of device memory with `hipMalloc`. Zero-initialized (calloc contract).
    pub fn alloc(bytes: usize, _stream: ffi::hipStream_t) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if bytes > 0 {
            let rc = unsafe { ffi::hipMalloc(&mut ptr, bytes) };
            if rc != HIP_SUCCESS {
                panic!("hipMalloc({bytes}): rc={rc}");
            }
            // Zero-init (calloc contract)
            unsafe { ffi::hipMemset(ptr, 0, bytes) };
        }
        Self {
            ptr,
            len: bytes,
            owned: true,
        }
    }

    /// Allocate zero-initialized device memory (calloc contract via hipMalloc + hipMemset).
    pub fn alloc_zero(bytes: usize, stream: ffi::hipStream_t) -> Self {
        Self::alloc(bytes, stream)
    }

    /// Upload host bytes to this device buffer.
    pub fn upload(&mut self, src: &[u8], _stream: ffi::hipStream_t) {
        if src.is_empty() || self.ptr.is_null() {
            return;
        }
        let n = src.len().min(self.len);
        let rc = unsafe {
            ffi::hipMemcpy(
                self.ptr,
                src.as_ptr() as *const c_void,
                n,
                HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if rc != HIP_SUCCESS {
            panic!("hipMemcpy H2D: rc={rc}");
        }
    }

    /// Download device bytes to host.
    pub fn download(&self, dst: &mut [u8], stream: ffi::hipStream_t) {
        if dst.is_empty() || self.ptr.is_null() {
            return;
        }
        let n = dst.len().min(self.len);
        let rc = unsafe {
            ffi::hipMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                n,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if rc != HIP_SUCCESS {
            panic!("hipMemcpy D2H: rc={rc}");
        }
        // Wait for the copy to finish
        unsafe { ffi::hipStreamSynchronize(stream) };
    }
}

impl Buffer for RocmBuffer {
    fn len_bytes(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Drop for RocmBuffer {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe { ffi::hipFree(self.ptr) };
        }
    }
}

// ── RocmBackend ──────────────────────────────────────────────────────────────

/// The ROCm/HIP compute backend.
pub struct RocmBackend {
    /// Active device index.
    device: c_int,
    /// Non-blocking stream for all work.
    stream: ffi::hipStream_t,
    /// Compiled kernel module + function cache.
    pipelines: Pipelines,
    /// Dequantized-weight cache: bound-buffer device address → f16 device buffer.
    /// Single-generation lifetime (one backend per generation); keys are stable.
    pub(crate) weight_cache: Mutex<std::collections::HashMap<usize, RocmBuffer>>,
    /// Active weight-load progress bar.
    weight_pb: Arc<Mutex<Option<indicatif::ProgressBar>>>,
}

// The backend owns streams and device handles which are Send/Sync.
unsafe impl Send for RocmBackend {}
unsafe impl Sync for RocmBackend {}

impl RocmBackend {
    /// Create a new ROCm backend on the given device index.
    pub fn new(device_id: c_int) -> Result<Self> {
        let mut count: c_int = 0;
        let rc = unsafe { ffi::hipGetDeviceCount(&mut count) };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipGetDeviceCount: rc={rc}")));
        }
        if count == 0 {
            return Err(be("no HIP-capable devices found"));
        }
        if device_id >= count {
            return Err(be(format!(
                "HIP device {device_id} out of range (count={count})"
            )));
        }

        let device: c_int = device_id;
        let rc = unsafe { ffi::hipSetDevice(device) };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipSetDevice({device}): rc={rc}")));
        }

        let mut stream: ffi::hipStream_t = std::ptr::null_mut();
        let rc = unsafe { ffi::hipStreamCreate(&mut stream) };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipStreamCreate: rc={rc}")));
        }

        let pipelines = Pipelines::build(device)?;

        Ok(Self {
            device,
            stream,
            pipelines,
            weight_cache: Mutex::new(std::collections::HashMap::new()),
            weight_pb: Arc::new(Mutex::new(None)),
        })
    }

    /// Read a device property field.
    fn prop(&self) -> ffi::hipDeviceProp_t {
        let mut props: ffi::hipDeviceProp_t = unsafe { std::mem::zeroed() };
        unsafe { ffi::hipGetDeviceProperties(&mut props, self.device) };
        props
    }
}

impl Backend for RocmBackend {
    fn name(&self) -> &str {
        "rocm"
    }

    fn capabilities(&self) -> Capabilities {
        let props = self.prop();
        Capabilities {
            name: "AMD ROCm/HIP".into(),
            f16: true,
            coopmat_f16: None,
            f8: false,
            coopmat_f8: None,
            i8: false,
            i8_dot: false,
            coopmat_i8: None,
            bf16: false,
            coopmat_bf16: None,
            subgroup_min: 0,
            subgroup_max: 0,
            sg_pref: 0,
            vendor_intel: false,
            integrated: false,
            compute_units: props.multi_processor_count as u32,
            buffer_device_address: false,
            max_shared_memory_bytes: props.shared_mem_per_block as u32,
            unified_memory: false,
            // ── correctness-dial: start with NOTHING fused ──
            decode_replay: false,
            combined_gu: false,
            embed_gather: false,
            gpu_sample: false,
            argmax_rows: false,
            argmax_prob: false,
            gated_rmsnorm: false,
            kv_swa_ring: false,
        }
    }

    fn alloc(&self, bytes: usize, _usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        let buf = RocmBuffer::alloc(bytes, self.stream);
        // Advance weight progress bar for weight/host-weight allocations
        if matches!(_usage, BufferUsage::Weights | BufferUsage::HostWeights) {
            if let Some(pb) = self.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
        }
        Ok(Box::new(buf))
    }

    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // Skip zero-init for weight buffers (they get uploaded immediately).
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if bytes > 0 {
            let rc = unsafe { ffi::hipMalloc(&mut ptr, bytes) };
            if rc != HIP_SUCCESS {
                return Err(be(format!("hipMalloc: rc={rc}")));
            }
        }
        if matches!(usage, BufferUsage::Weights | BufferUsage::HostWeights) {
            if let Some(pb) = self.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
        }
        Ok(Box::new(RocmBuffer {
            ptr,
            len: bytes,
            owned: true,
        }))
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        let buf = dst
            .as_any()
            .downcast_ref::<RocmBuffer>()
            .expect("rocm backend: buffer is not a RocmBuffer");
        if src.is_empty() || buf.ptr.is_null() {
            return Ok(());
        }
        let n = src.len().min(buf.len);
        let rc = unsafe {
            ffi::hipMemcpy(
                buf.ptr,
                src.as_ptr() as *const c_void,
                n,
                HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipMemcpy H2D: rc={rc}")));
        }
        Ok(())
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        let buf = src
            .as_any()
            .downcast_ref::<RocmBuffer>()
            .expect("rocm backend: buffer is not a RocmBuffer");
        if dst.is_empty() || buf.ptr.is_null() {
            return Ok(());
        }
        let n = dst.len().min(buf.len);
        let rc = unsafe {
            ffi::hipMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                buf.ptr,
                n,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipMemcpy D2H: rc={rc}")));
        }
        // Wait for download to complete
        unsafe { ffi::hipStreamSynchronize(self.stream) };
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        Ok(GraphPlan::boxed(graph))
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        exec::execute_graph(
            &self.pipelines,
            &self.weight_cache,
            self.stream,
            plan,
            bindings,
        )
    }

    fn sync(&self) -> Result<()> {
        let rc = unsafe { ffi::hipStreamSynchronize(self.stream) };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipStreamSynchronize: rc={rc}")));
        }
        Ok(())
    }

    fn copy_buffer(&self, src: &dyn Buffer, dst: &dyn Buffer, bytes: usize) -> Result<()> {
        infr_core::backend::check_copy_bytes(bytes, src.len_bytes())?;
        let src_buf = src
            .as_any()
            .downcast_ref::<RocmBuffer>()
            .expect("rocm backend: src is not a RocmBuffer");
        let dst_buf = dst
            .as_any()
            .downcast_ref::<RocmBuffer>()
            .expect("rocm backend: dst is not a RocmBuffer");
        let rc = unsafe { ffi::hipMemcpyDtoD(dst_buf.ptr, src_buf.ptr, bytes) };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipMemcpyDtoD: rc={rc}")));
        }
        Ok(())
    }

    fn weight_progress(&self, total_bytes: Option<u64>) -> Box<dyn ProgressScope> {
        struct RocmProgress {
            pb: Arc<Mutex<Option<indicatif::ProgressBar>>>,
        }
        impl ProgressScope for RocmProgress {}
        impl Drop for RocmProgress {
            fn drop(&mut self) {
                if let Some(pb) = self.pb.lock().unwrap().take() {
                    pb.finish_and_clear();
                }
            }
        }
        let pb = total_bytes.map(|total| {
            let style = indicatif::ProgressStyle::with_template(
                "  {spinner} ROCm weights {bytes}/{total_bytes} [{elapsed_precise}] {msg}",
            )
            .unwrap();
            let pb = indicatif::ProgressBar::new(total);
            pb.set_style(style);
            pb
        });
        *self.weight_pb.lock().unwrap() = pb;
        Box::new(RocmProgress {
            pb: self.weight_pb.clone(),
        })
    }
}

impl Drop for RocmBackend {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                ffi::hipStreamSynchronize(self.stream);
                ffi::hipStreamDestroy(self.stream);
            }
        }
    }
}
