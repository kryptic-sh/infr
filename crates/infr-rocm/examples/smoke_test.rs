//! Smoke test: validate the ROCm backend constructs, allocates, and runs a trivial kernel.
//! Build: `cargo build --release --features rocm -p infr-rocm --example smoke_test`
//! Run: `LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/examples/smoke_test`

use std::ffi::{c_char, c_int, c_void, CString};

// Minimal FFI subset
type hipModule_t = *mut c_void;
type hipFunction_t = *mut c_void;
type hiprtcProgram = *mut c_void;

#[link(name = "amdhip64")]
extern "C" {
    fn hipGetDeviceCount(count: *mut c_int) -> c_int;
    fn hipSetDevice(device: c_int) -> c_int;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn hipFree(ptr: *mut c_void) -> c_int;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn hipMemset(dst: *mut c_void, value: c_int, count: usize) -> c_int;
    fn hipStreamCreate(stream: *mut *mut c_void) -> c_int;
    fn hipStreamSynchronize(stream: *mut c_void) -> c_int;
    fn hipStreamDestroy(stream: *mut c_void) -> c_int;
    fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> c_int;
    fn hipModuleGetFunction(
        func: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> c_int;
    fn hipModuleLaunchKernel(
        f: hipFunction_t,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shm: u32,
        stream: *mut c_void,
        kp: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;
}
#[link(name = "hiprtc")]
extern "C" {
    fn hiprtcCreateProgram(
        p: *mut hiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        nh: c_int,
        hdrs: *const *const c_char,
        incl: *const *const c_char,
    ) -> c_int;
    fn hiprtcCompileProgram(p: hiprtcProgram, no: c_int, opts: *const *const c_char) -> c_int;
    fn hiprtcGetCodeSize(p: hiprtcProgram, s: *mut usize) -> c_int;
    fn hiprtcGetCode(p: hiprtcProgram, code: *mut c_char) -> c_int;
    fn hiprtcGetProgramLogSize(p: hiprtcProgram, s: *mut usize) -> c_int;
    fn hiprtcGetProgramLog(p: hiprtcProgram, log: *mut c_char) -> c_int;
    fn hiprtcDestroyProgram(p: *mut hiprtcProgram) -> c_int;
}

const HIP_SUCCESS: c_int = 0;
const HIPRTC_SUCCESS: c_int = 0;
const H2D: c_int = 1;
const D2H: c_int = 2;

fn check(rc: c_int, msg: &str) {
    if rc != HIP_SUCCESS && rc != HIPRTC_SUCCESS {
        panic!("{msg}: rc={rc}");
    }
}

fn main() {
    // 1. Init device + stream
    let mut count: c_int = 0;
    check(
        unsafe { hipGetDeviceCount(&mut count) },
        "hipGetDeviceCount",
    );
    println!("Devices: {count}");
    assert!(count > 0);
    check(unsafe { hipSetDevice(0) }, "hipSetDevice");
    let mut stream: *mut c_void = std::ptr::null_mut();
    check(unsafe { hipStreamCreate(&mut stream) }, "hipStreamCreate");

    // 2. Compile a minimal kernel via hiprtc
    let kernel_src = r#"
extern "C" __global__ void add_one(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1.0f;
}
"#;
    let csrc = CString::new(kernel_src).unwrap();
    let cname = CString::new("test").unwrap();
    let mut prog: hiprtcProgram = std::ptr::null_mut();
    check(
        unsafe {
            hiprtcCreateProgram(
                &mut prog,
                csrc.as_ptr(),
                cname.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        },
        "hiprtcCreateProgram",
    );

    let std_flag = CString::new("-std=c++17").unwrap();
    let opts: [*const c_char; 1] = [std_flag.as_ptr()];
    check(
        unsafe { hiprtcCompileProgram(prog, 1, opts.as_ptr()) },
        "hiprtcCompileProgram",
    );

    let mut code_size: usize = 0;
    check(
        unsafe { hiprtcGetCodeSize(prog, &mut code_size) },
        "hiprtcGetCodeSize",
    );
    println!("Kernel code size: {code_size} bytes");
    let mut code: Vec<u8> = vec![0; code_size];
    check(
        unsafe { hiprtcGetCode(prog, code.as_mut_ptr() as *mut c_char) },
        "hiprtcGetCode",
    );
    unsafe { hiprtcDestroyProgram(&mut prog) };

    // 3. Load module + get function
    let mut module: hipModule_t = std::ptr::null_mut();
    check(
        unsafe { hipModuleLoadData(&mut module, code.as_ptr() as *const c_void) },
        "hipModuleLoadData",
    );
    let cfn = CString::new("add_one").unwrap();
    let mut func: hipFunction_t = std::ptr::null_mut();
    check(
        unsafe { hipModuleGetFunction(&mut func, module, cfn.as_ptr()) },
        "hipModuleGetFunction",
    );

    // 4. Alloc GPU buffer, upload data, launch kernel, download
    let n: i32 = 1024;
    let mut dptr: *mut c_void = std::ptr::null_mut();
    check(
        unsafe { hipMalloc(&mut dptr, (n as usize) * 4) },
        "hipMalloc",
    );
    let host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    check(
        unsafe { hipMemcpy(dptr, host.as_ptr() as *const c_void, host.len() * 4, H2D) },
        "hipMemcpy H2D",
    );

    let block_size: u32 = 256;
    let grid_size: u32 = ((n as u32) + block_size - 1) / block_size;
    let n_arg = n;
    let mut args: [*mut c_void; 2] = [
        &mut dptr as *mut _ as *mut c_void,
        &n_arg as *const i32 as *mut c_void,
    ];
    check(
        unsafe {
            hipModuleLaunchKernel(
                func,
                grid_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        "hipModuleLaunchKernel",
    );

    check(
        unsafe { hipStreamSynchronize(stream) },
        "hipStreamSynchronize",
    );

    let mut result: Vec<f32> = vec![0.0; n as usize];
    check(
        unsafe {
            hipMemcpy(
                result.as_mut_ptr() as *mut c_void,
                dptr,
                result.len() * 4,
                D2H,
            )
        },
        "hipMemcpy D2H",
    );
    check(unsafe { hipStreamSynchronize(stream) }, "sync");

    // 5. Verify: each element should be original + 1
    let mut ok = true;
    for i in 0..n as usize {
        let expected = i as f32 + 1.0;
        if (result[i] - expected).abs() > 0.001 {
            println!("MISMATCH at {i}: got {}, expected {expected}", result[i]);
            ok = false;
        }
    }
    if ok {
        println!("PASS: all {n} elements correct (original + 1)");
    }

    // 6. Cleanup
    unsafe { hipFree(dptr) };
    unsafe { hipStreamDestroy(stream) };
}
