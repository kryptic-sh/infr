//! HIP FFI — hand-rolled `extern "C"` bindings to `libamdhip64` and `libhiprtc`.
//! Compiled only when `cfg(all(target_os = "linux", feature = "rocm"))`.

use std::ffi::{c_char, c_int, c_void};

#[link(name = "amdhip64")]
extern "C" {
    pub fn hipGetDeviceCount(count: *mut c_int) -> c_int;
    pub fn hipSetDevice(device: c_int) -> c_int;
    pub fn hipGetDeviceProperties(props: *mut hipDeviceProp_t, device: c_int) -> c_int;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    pub fn hipFree(ptr: *mut c_void) -> c_int;
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: hipMemcpyKind) -> c_int;
    pub fn hipMemset(dst: *mut c_void, value: c_int, count: usize) -> c_int;
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> c_int;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> c_int;
    pub fn hipStreamDestroy(stream: hipStream_t) -> c_int;
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> c_int;
    pub fn hipModuleGetFunction(function: *mut hipFunction_t, module: hipModule_t, name: *const c_char) -> c_int;
    #[allow(improper_ctypes)]
    pub fn hipModuleLaunchKernel(f: hipFunction_t, grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32, block_dim_x: u32, block_dim_y: u32, block_dim_z: u32, shared_mem_bytes: u32, stream: hipStream_t, kernel_params: *mut *mut c_void, extra: *mut *mut c_void) -> c_int;
    pub fn hipDeviceSynchronize() -> c_int;
    pub fn hipMemcpyDtoD(dst: *mut c_void, src: *const c_void, count: usize) -> c_int;
}

#[link(name = "hiprtc")]
extern "C" {
    pub fn hiprtcCreateProgram(prog: *mut hiprtcProgram, src: *const c_char, name: *const c_char, num_headers: c_int, headers: *const *const c_char, include_names: *const *const c_char) -> c_int;
    pub fn hiprtcCompileProgram(prog: hiprtcProgram, num_options: c_int, options: *const *const c_char) -> c_int;
    pub fn hiprtcGetCode(prog: hiprtcProgram, code: *mut c_char) -> c_int;
    pub fn hiprtcGetCodeSize(prog: hiprtcProgram, size: *mut usize) -> c_int;
    pub fn hiprtcGetProgramLog(prog: hiprtcProgram, log: *mut c_char) -> c_int;
    pub fn hiprtcGetProgramLogSize(prog: hiprtcProgram, log_size: *mut usize) -> c_int;
    pub fn hiprtcDestroyProgram(prog: *mut hiprtcProgram) -> c_int;
}

pub type hipStream_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hiprtcProgram = *mut c_void;

pub const HIP_SUCCESS: c_int = 0;
pub const HIPRTC_SUCCESS: c_int = 0;

pub type hipMemcpyKind = c_int;
pub const HIP_MEMCPY_HOST_TO_DEVICE: hipMemcpyKind = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: hipMemcpyKind = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: hipMemcpyKind = 3;

#[repr(C)]
pub struct hipDeviceProp_t {
    pub name: [c_char; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub max_threads_per_block: c_int,
    pub max_threads_per_multi_processor: c_int,
    pub multi_processor_count: c_int,
    pub clock_rate: c_int,
    pub memory_clock_rate: c_int,
    pub memory_bus_width: c_int,
    pub l2_cache_size: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub compute_mode: c_int,
    pub major: c_int,
    pub minor: c_int,
    pub arch: hipDeviceArch_t,
    pub gcn_arch: c_int,
    pub gcn_arch_name: [c_char; 256],
    pub _pad: [u8; 128],
}

#[repr(C)]
pub struct hipDeviceArch_t {
    pub has_watchdog: c_int,
    pub cooperative_launch: c_int,
}
