//! Quick test: compile a trivial HIP kernel via hiprtc to verify the toolchain.

// A throwaway bringup tool with its own hand-rolled hiprtc FFI: keep the C type names and the
// full binding surface even though some entries are unused here.
#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr;

type hiprtcProgram = *mut c_void;
type hiprtcResult = c_int;

const HIPRTC_SUCCESS: c_int = 0;

#[link(name = "hiprtc")]
extern "C" {
    fn hiprtcCreateProgram(
        prog: *mut hiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> hiprtcResult;
    fn hiprtcCompileProgram(
        prog: hiprtcProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> hiprtcResult;
    fn hiprtcGetProgramLogSize(prog: hiprtcProgram, size: *mut usize) -> hiprtcResult;
    fn hiprtcGetProgramLog(prog: hiprtcProgram, log: *mut c_char) -> hiprtcResult;
    fn hiprtcGetCodeSize(prog: hiprtcProgram, size: *mut usize) -> hiprtcResult;
    fn hiprtcGetCode(prog: hiprtcProgram, code: *mut c_char) -> hiprtcResult;
    fn hiprtcDestroyProgram(prog: *mut hiprtcProgram) -> hiprtcResult;
}

fn main() {
    let src = r#"
extern "C" __global__ void add_one(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1.0f;
}
"#;
    let csrc = CString::new(src).unwrap();
    let cname = CString::new("test_kern").unwrap();
    let mut prog: hiprtcProgram = ptr::null_mut();

    let rc = unsafe {
        hiprtcCreateProgram(
            &mut prog,
            csrc.as_ptr(),
            cname.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        )
    };
    println!("CreateProgram: rc={rc}");
    assert_eq!(rc, HIPRTC_SUCCESS);

    let opts: Vec<CString> = vec![
        // CString::new(format!("--gpu-architecture=gfx1100")),
    ];
    let opt_ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
    let opt_ptrs_ptr = if opt_ptrs.is_empty() {
        ptr::null()
    } else {
        opt_ptrs.as_ptr()
    };

    let rc = unsafe { hiprtcCompileProgram(prog, opt_ptrs.len() as c_int, opt_ptrs_ptr) };
    println!("CompileProgram: rc={rc}");

    if rc != HIPRTC_SUCCESS {
        let mut log_size: usize = 0;
        unsafe { hiprtcGetProgramLogSize(prog, &mut log_size) };
        println!("Log size: {log_size}");
        let mut log_buf: Vec<u8> = vec![0; log_size.max(1)];
        unsafe { hiprtcGetProgramLog(prog, log_buf.as_mut_ptr() as *mut c_char) };
        let log = String::from_utf8_lossy(&log_buf);
        println!("Compile log:\n===START===\n{log}\n===END===");
    }

    unsafe { hiprtcDestroyProgram(&mut prog) };
}
