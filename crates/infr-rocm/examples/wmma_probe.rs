//! WMMA de-risking probe: does hiprtc (auto-detect gfx1100) compile and correctly run the
//! RDNA3 wave32 matrix-core builtins? Validates both the f16→f32 path
//! (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`) and the int8→i32 path
//! (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`) on a single 16x16x16 tile against a host
//! reference, so the fragment layout is confirmed before the real prefill GEMM is written.
//!
//! Build: cargo build --features rocm -p infr-rocm --example wmma_probe
//! Run:   LD_LIBRARY_PATH=/opt/rocm/lib ./target/debug/examples/wmma_probe

#![allow(non_camel_case_types, dead_code, clippy::needless_range_loop)]

use std::ffi::{c_char, c_int, c_void, CString};

type hipModule_t = *mut c_void;
type hipFunction_t = *mut c_void;
type hiprtcProgram = *mut c_void;

#[link(name = "amdhip64")]
extern "C" {
    fn hipSetDevice(device: c_int) -> c_int;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn hipFree(ptr: *mut c_void) -> c_int;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn hipStreamCreate(stream: *mut *mut c_void) -> c_int;
    fn hipStreamSynchronize(stream: *mut c_void) -> c_int;
    fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> c_int;
    fn hipModuleGetFunction(func: *mut hipFunction_t, m: hipModule_t, n: *const c_char) -> c_int;
    #[allow(clippy::too_many_arguments)]
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

const H2D: c_int = 1;
const D2H: c_int = 2;

fn ck(rc: c_int, m: &str) {
    assert_eq!(rc, 0, "{m}: rc={rc}");
}

const SRC: &str = r#"
typedef __fp16   half16 __attribute__((ext_vector_type(16)));
typedef float    float8 __attribute__((ext_vector_type(8)));
typedef int      int4v  __attribute__((ext_vector_type(4)));
typedef int      int8v  __attribute__((ext_vector_type(8)));

// D[16x16] = A[16x16(MxK)] @ B[16x16(KxN)], f16 in / f32 accumulate. One wave32 block.
extern "C" __global__ void wmma_f16(const __fp16* A, const __fp16* B, float* D) {
    int lane = threadIdx.x;         // 0..31
    int r = lane % 16;              // A row / B col this lane feeds
    half16 a, b;
    for (int k = 0; k < 16; k++) a[k] = A[r * 16 + k];   // A[row=r][k]
    for (int k = 0; k < 16; k++) b[k] = B[k * 16 + r];   // B[k][col=r]
    float8 c = {0,0,0,0,0,0,0,0};
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    for (int e = 0; e < 8; e++) {
        int row = e * 2 + (lane / 16);
        int col = lane % 16;
        D[row * 16 + col] = c[e];
    }
}

// D[16x16] = A(int8 MxK) @ B(int8 KxN), i32 accumulate. int8 packed 4/lane-element (16 K / lane).
extern "C" __global__ void wmma_iu8(const int* A, const int* B, int* D) {
    int lane = threadIdx.x;
    int r = lane % 16;
    int4v a, b;
    // A row-major [16][16] int8 → 4 ints/row; lane's row = r.
    for (int k = 0; k < 4; k++) a[k] = A[r * 4 + k];
    // B needs column r's 16 K values packed. B stored column-major-packed: Bpack[col*4 + kb].
    for (int k = 0; k < 4; k++) b[k] = B[r * 4 + k];
    int8v c = {0,0,0,0,0,0,0,0};
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
    for (int e = 0; e < 8; e++) {
        int row = e * 2 + (lane / 16);
        int col = lane % 16;
        D[row * 16 + col] = c[e];
    }
}
"#;

fn compile() -> Vec<u8> {
    let csrc = CString::new(SRC).unwrap();
    let cname = CString::new("probe").unwrap();
    let mut prog: hiprtcProgram = std::ptr::null_mut();
    ck(
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
        "create",
    );
    let f = CString::new("-std=c++17").unwrap();
    let opts = [f.as_ptr()];
    let rc = unsafe { hiprtcCompileProgram(prog, 1, opts.as_ptr()) };
    if rc != 0 {
        let mut sz = 0usize;
        unsafe { hiprtcGetProgramLogSize(prog, &mut sz) };
        let mut buf = vec![0u8; sz.max(1)];
        unsafe { hiprtcGetProgramLog(prog, buf.as_mut_ptr() as *mut c_char) };
        panic!("compile rc={rc}:\n{}", String::from_utf8_lossy(&buf));
    }
    let mut sz = 0usize;
    ck(unsafe { hiprtcGetCodeSize(prog, &mut sz) }, "codesize");
    let mut code = vec![0u8; sz];
    ck(
        unsafe { hiprtcGetCode(prog, code.as_mut_ptr() as *mut c_char) },
        "getcode",
    );
    unsafe { hiprtcDestroyProgram(&mut prog) };
    code
}

fn dev_alloc(bytes: usize) -> *mut c_void {
    let mut p: *mut c_void = std::ptr::null_mut();
    ck(unsafe { hipMalloc(&mut p, bytes) }, "malloc");
    p
}

fn main() {
    ck(unsafe { hipSetDevice(0) }, "setdev");
    let mut stream: *mut c_void = std::ptr::null_mut();
    ck(unsafe { hipStreamCreate(&mut stream) }, "stream");

    let code = compile();
    println!("hiprtc compiled WMMA module OK ({} bytes)", code.len());
    let mut module: hipModule_t = std::ptr::null_mut();
    ck(
        unsafe { hipModuleLoadData(&mut module, code.as_ptr() as *const c_void) },
        "load",
    );

    // ── f16 path ──
    let a: Vec<f32> = (0..256)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
        .collect();
    let b: Vec<f32> = (0..256)
        .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1)
        .collect();
    let mut refd = vec![0f32; 256];
    for m in 0..16 {
        for n in 0..16 {
            let mut acc = 0f32;
            for k in 0..16 {
                acc += half::f16::from_f32(a[m * 16 + k]).to_f32()
                    * half::f16::from_f32(b[k * 16 + n]).to_f32();
            }
            refd[m * 16 + n] = acc;
        }
    }
    let a16: Vec<u16> = a
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let b16: Vec<u16> = b
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let da = dev_alloc(512);
    let db = dev_alloc(512);
    let dd = dev_alloc(1024);
    ck(
        unsafe { hipMemcpy(da, a16.as_ptr() as *const c_void, 512, H2D) },
        "h2d a",
    );
    ck(
        unsafe { hipMemcpy(db, b16.as_ptr() as *const c_void, 512, H2D) },
        "h2d b",
    );
    let cfn = CString::new("wmma_f16").unwrap();
    let mut func: hipFunction_t = std::ptr::null_mut();
    ck(
        unsafe { hipModuleGetFunction(&mut func, module, cfn.as_ptr()) },
        "getfn f16",
    );
    let mut pa = da;
    let mut pb = db;
    let mut pd = dd;
    let mut args: [*mut c_void; 3] = [
        &mut pa as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pd as *mut _ as *mut c_void,
    ];
    ck(
        unsafe {
            hipModuleLaunchKernel(
                func,
                1,
                1,
                1,
                32,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        "launch f16",
    );
    ck(unsafe { hipStreamSynchronize(stream) }, "sync f16");
    let mut got = vec![0f32; 256];
    ck(
        unsafe { hipMemcpy(got.as_mut_ptr() as *mut c_void, dd, 1024, D2H) },
        "d2h f16",
    );
    let mut maxe = 0f32;
    for i in 0..256 {
        maxe = maxe.max((got[i] - refd[i]).abs());
    }
    let refmag = refd.iter().fold(0f32, |m, &v| m.max(v.abs()));
    println!(
        "WMMA f16→f32: max_err={maxe:e} max|ref|={refmag:e}  {}",
        if maxe < 1e-2 { "PASS" } else { "FAIL" }
    );

    // ── int8 path ── A row-major int8 [16][16]; B packed by column: Bp[col*4+kb] holds K 4kb..4kb+3.
    let ones = std::env::var_os("ONES").is_some();
    let ai: Vec<i8> = if ones {
        vec![1i8; 256]
    } else {
        (0..256).map(|i| (i * 7 % 17 - 8) as i8).collect()
    };
    let bi: Vec<i8> = if ones {
        vec![1i8; 256]
    } else {
        (0..256).map(|i| (i * 5 % 15 - 7) as i8).collect()
    }; // B[k][n], row-major
    let mut refi = vec![0i32; 256];
    for m in 0..16 {
        for n in 0..16 {
            let mut acc = 0i32;
            for k in 0..16 {
                acc += ai[m * 16 + k] as i32 * bi[k * 16 + n] as i32;
            }
            refi[m * 16 + n] = acc;
        }
    }
    // Pack A: row-major already, 4 ints per row (16 int8). Pack B: column n's 16 K values → 4 ints.
    let pack = |bytes: [i8; 4]| -> i32 {
        (bytes[0] as u8 as i32)
            | ((bytes[1] as u8 as i32) << 8)
            | ((bytes[2] as u8 as i32) << 16)
            | ((bytes[3] as u8 as i32) << 24)
    };
    let mut apk = vec![0i32; 64];
    for row in 0..16 {
        for kb in 0..4 {
            apk[row * 4 + kb] = pack([
                ai[row * 16 + kb * 4],
                ai[row * 16 + kb * 4 + 1],
                ai[row * 16 + kb * 4 + 2],
                ai[row * 16 + kb * 4 + 3],
            ]);
        }
    }
    let mut bpk = vec![0i32; 64];
    for col in 0..16 {
        for kb in 0..4 {
            bpk[col * 4 + kb] = pack([
                bi[(kb * 4) * 16 + col],
                bi[(kb * 4 + 1) * 16 + col],
                bi[(kb * 4 + 2) * 16 + col],
                bi[(kb * 4 + 3) * 16 + col],
            ]);
        }
    }
    let da2 = dev_alloc(256);
    let db2 = dev_alloc(256);
    let dd2 = dev_alloc(1024);
    ck(
        unsafe { hipMemcpy(da2, apk.as_ptr() as *const c_void, 256, H2D) },
        "h2d ai",
    );
    ck(
        unsafe { hipMemcpy(db2, bpk.as_ptr() as *const c_void, 256, H2D) },
        "h2d bi",
    );
    let cfn2 = CString::new("wmma_iu8").unwrap();
    let mut func2: hipFunction_t = std::ptr::null_mut();
    ck(
        unsafe { hipModuleGetFunction(&mut func2, module, cfn2.as_ptr()) },
        "getfn iu8",
    );
    let mut pa2 = da2;
    let mut pb2 = db2;
    let mut pd2 = dd2;
    let mut args2: [*mut c_void; 3] = [
        &mut pa2 as *mut _ as *mut c_void,
        &mut pb2 as *mut _ as *mut c_void,
        &mut pd2 as *mut _ as *mut c_void,
    ];
    ck(
        unsafe {
            hipModuleLaunchKernel(
                func2,
                1,
                1,
                1,
                32,
                1,
                1,
                0,
                stream,
                args2.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        "launch iu8",
    );
    ck(unsafe { hipStreamSynchronize(stream) }, "sync iu8");
    let mut goti = vec![0i32; 256];
    ck(
        unsafe { hipMemcpy(goti.as_mut_ptr() as *mut c_void, dd2, 1024, D2H) },
        "d2h iu8",
    );
    let mut mism = 0;
    for i in 0..256 {
        if goti[i] != refi[i] {
            if mism < 4 {
                println!("  iu8 mismatch @{i}: got={} ref={}", goti[i], refi[i]);
            }
            mism += 1;
        }
    }
    println!(
        "WMMA iu8→i32: mismatches={mism}/256  {}",
        if mism == 0 { "PASS" } else { "FAIL" }
    );
    if ones {
        println!("  ONES: got[0..8]={:?}  (expect all 16)", &goti[0..8]);
        println!("  ONES: got row1 [16..24]={:?}", &goti[16..24]);
    }

    unsafe {
        hipFree(da);
        hipFree(db);
        hipFree(dd);
        hipFree(da2);
        hipFree(db2);
        hipFree(dd2);
    }
}
