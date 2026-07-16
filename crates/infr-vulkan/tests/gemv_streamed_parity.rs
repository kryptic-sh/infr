//! Proves each decode-GEMV kernel's `-DSTREAMED` twin (weight read through a
//! `bufferDeviceAddress` arena pointer) computes BIT-IDENTICALLY to its resident twin (weight read
//! through a bound SSBO). The two builds differ only in where `NW(i)` sources its word from — the
//! dequant math and the accumulation order in `native_decode.glsl` are the same code — so anything
//! short of bitwise equality is a bug, not a tolerance question.
//!
//! The second, load-bearing assertion is the NON-ZERO ARENA OFFSET: the resident-BDA integration
//! places every tensor at its own byte offset inside ONE multi-GiB arena and passes
//! `arena_addr + tensor_off` as the kernel's base. A twin that only works at offset 0 would pass a
//! naive parity test and then read garbage for every tensor but the first, so each case is run
//! twice — once at offset 0, once at a non-zero offset — and both must match the resident result.
//!
//! WEIGHT BYTES: deliberately NOT a faithful quantization of any target tensor. Parity only needs
//! both paths to decode the SAME bytes; what float those bytes mean is irrelevant. Every byte is
//! drawn from `0x00..=0x3F`, which is what makes this safe for EVERY dtype without hand-rolling a
//! per-format quantizer: an f16 is NaN/Inf only when its 5 exponent bits are all ones, which needs
//! bit 14 (0x40) set in the high byte — unreachable when every byte is < 0x40. So all scales
//! decode finite and non-degenerate, no output is NaN (which would make a bitwise compare pass
//! vacuously, hiding the very mis-addressing this test exists to catch), and a wrong address still
//! yields visibly different finite floats.
//!
//! Run: `cargo test -p infr-vulkan --test gemv_streamed_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

/// Pseudo-random weight bytes in `0x00..=0x3F` — see the module header for why the range matters.
fn synth_weight_bytes(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7;
            (h % 0x40) as u8
        })
        .collect()
}

/// Activations kept small and mixed-sign so a sign/stride error shows up as a magnitude change.
fn synth_x(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect()
}

fn weight_bytes_for(dtype: DType, elems: usize) -> usize {
    let (blk_elems, blk_bytes) = infr_gguf::block_layout(dtype);
    assert_eq!(
        elems % blk_elems,
        0,
        "test shape must be a whole number of {dtype:?} blocks"
    );
    elems / blk_elems * blk_bytes
}

/// Byte offset a tensor sits at inside the shared arena for the non-zero-offset leg. Must be a
/// whole number of blocks (so `w_base`-relative decode still lands on a block boundary) and is
/// 256-byte aligned to satisfy any natural alignment the pointer reads assume.
fn nonzero_off(dtype: DType) -> usize {
    let (_, blk_bytes) = infr_gguf::block_layout(dtype);
    let mut off = blk_bytes * 8;
    off = off.div_ceil(256) * 256;
    // Re-round UP to a block multiple; 256-alignment alone can land mid-block for odd block sizes.
    off.div_ceil(blk_bytes) * blk_bytes
}

struct Case {
    /// Human name for the failure message.
    name: String,
    /// Bits of the resident dispatch's output.
    resident: Vec<u32>,
    /// Bits of the streamed dispatch's output, at arena offset 0 and at a non-zero offset.
    streamed_at0: Vec<u32>,
    streamed_atoff: Vec<u32>,
}

fn bits(v: &[u8]) -> Vec<u32> {
    bytemuck::cast_slice::<u8, u32>(v).to_vec()
}

type Buf = dyn infr_core::backend::Buffer;
/// Records the resident dispatch: `(recorder, weight_ssbo, x, y)`.
type DispatchRes<'f> = dyn Fn(&infr_vulkan::Recorder, &Buf, &Buf, &Buf) + 'f;
/// Records the streamed dispatch: `(recorder, weight_device_address, x, y)`.
type DispatchStr<'f> = dyn Fn(&infr_vulkan::Recorder, u64, &Buf, &Buf) + 'f;

/// Runs one kernel family through resident + both streamed legs. `dispatch_res` records the
/// resident dispatch; `dispatch_str` records the streamed one against a given device address.
#[allow(clippy::too_many_arguments)]
fn run_case(
    be: &VulkanBackend,
    name: String,
    dtype: DType,
    in_f: usize,
    out_f: usize,
    out_elems: usize,
    dispatch_res: &DispatchRes,
    dispatch_str: &DispatchStr,
) -> Case {
    let w_elems = in_f * out_f;
    let w_bytes = weight_bytes_for(dtype, w_elems);
    let w = synth_weight_bytes(w_bytes, in_f * 31 + out_f);

    let x = synth_x(in_f);
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let y_buf = be.alloc(out_elems * 4, BufferUsage::Activations).unwrap();

    // ── Resident: weight in its own bound SSBO ────────────────────────────────────────────────
    let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_res(&rec, w_buf.as_ref(), x_buf.as_ref(), y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let resident = bits(&out);

    // ── Streamed leg 1: arena offset 0 ────────────────────────────────────────────────────────
    let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena0.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr0, x_buf.as_ref(), y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let streamed_at0 = bits(&out);

    // ── Streamed leg 2: the SAME weight parked at a non-zero offset in a bigger arena ─────────
    // Mirrors the resident-BDA layout (one arena, many tensors). The prefix is filled with
    // DIFFERENT bytes, so a twin that ignores the offset and reads from the arena base produces a
    // visibly wrong result instead of accidentally matching.
    let off = nonzero_off(dtype);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);
    let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena1.as_ref(), &backing).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr1 + off as u64, x_buf.as_ref(), y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let streamed_atoff = bits(&out);

    Case {
        name,
        resident,
        streamed_at0,
        streamed_atoff,
    }
}

fn assert_case(c: &Case) {
    assert!(
        c.resident.iter().any(|&b| b != 0),
        "{}: resident output is all zeros — the case is not exercising the kernel",
        c.name
    );
    for (i, (&r, &s)) in c.resident.iter().zip(c.streamed_at0.iter()).enumerate() {
        assert_eq!(
            r,
            s,
            "{}: streamed@0 differs from resident at out {i}: {} vs {} (bits {r:#010x} vs {s:#010x})",
            c.name,
            f32::from_bits(r),
            f32::from_bits(s)
        );
    }
    for (i, (&r, &s)) in c.resident.iter().zip(c.streamed_atoff.iter()).enumerate() {
        assert_eq!(
            r,
            s,
            "{}: streamed@nonzero-offset differs from resident at out {i}: {} vs {} — the twin is \
             ignoring its arena base offset, which breaks every tensor but the first in a shared arena",
            c.name,
            f32::from_bits(r),
            f32::from_bits(s)
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mrow_streamed_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    for dtype in [DType::Q8_0, DType::Q4K, DType::Q6K] {
        for rows in [2usize, 4, 8] {
            let c = run_case(
                &be,
                format!("mrow dtype={dtype:?} rows={rows}"),
                dtype,
                in_f,
                out_f,
                rows * out_f,
                &|rec, w, x, y| rec.linear_native_mrow(dtype, w, 0, x, y, rows, in_f, out_f),
                &|rec, addr, x, y| {
                    rec.linear_native_mrow_streamed(dtype, addr, 0, x, y, rows, in_f, out_f)
                },
            );
            assert_case(&c);
            println!("ok: {}", c.name);
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn rm_streamed_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    for dtype in [DType::Q4K, DType::Q6K] {
        for rm in [2u32, 4] {
            let c = run_case(
                &be,
                format!("rm dtype={dtype:?} rm={rm}"),
                dtype,
                in_f,
                out_f,
                out_f,
                &|rec, w, x, y| rec.linear_native_rm(dtype, w, 0, x, y, in_f, out_f, rm),
                &|rec, addr, x, y| {
                    rec.linear_native_rm_streamed(dtype, addr, 0, x, y, in_f, out_f, rm)
                },
            );
            assert_case(&c);
            println!("ok: {}", c.name);
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn sg_streamed_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    let dtype = DType::Q6K; // the only dtype with an SG build
    for nr in [2u32, 4, 8] {
        let c = run_case(
            &be,
            format!("sg dtype={dtype:?} nr={nr}"),
            dtype,
            in_f,
            out_f,
            out_f,
            &|rec, w, x, y| rec.linear_native_sg(dtype, w, 0, x, y, in_f, out_f, nr),
            &|rec, addr, x, y| rec.linear_native_sg_streamed(dtype, addr, 0, x, y, in_f, out_f, nr),
        );
        assert_case(&c);
        println!("ok: {}", c.name);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn rm_v2_streamed_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    // Variant/dtype pairs that have a build (see gemm::native_rm_variant_spv).
    for (variant, dtype) in [
        ("sg", DType::Q4K),
        ("sg", DType::Q6K),
        ("dbuf", DType::Q4K),
        ("wg128", DType::Q4K),
        ("reg", DType::Q4K),
    ] {
        let c = run_case(
            &be,
            format!("rm_v2 variant={variant} dtype={dtype:?}"),
            dtype,
            in_f,
            out_f,
            out_f,
            &|rec, w, x, y| rec.linear_native_rm_v2(variant, dtype, w, 0, x, y, in_f, out_f),
            &|rec, addr, x, y| {
                rec.linear_native_rm_v2_streamed(variant, dtype, addr, 0, x, y, in_f, out_f)
            },
        );
        assert_case(&c);
        println!("ok: {}", c.name);
    }
}

/// The `w_base` within-tensor element offset must keep working under BDA — it is what lets one
/// stacked tensor (fused QKV slices, a stacked MoE expert bank) serve several logical weights, and
/// the resident-BDA layout composes it WITH the arena base offset (`arena_addr + tensor_off` as the
/// base, `w_base` as the slice within that tensor). Both must apply, so this checks a non-zero
/// `w_base` against a non-zero arena offset.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn w_base_composes_with_arena_offset() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let dtype = DType::Q4K;
    let (in_f, out_f) = (256usize, 8usize);
    // Two stacked "experts" in one tensor; the second is selected via w_base.
    let stride_elems = in_f * out_f;
    let w_bytes = weight_bytes_for(dtype, stride_elems * 2);
    let w = synth_weight_bytes(w_bytes, 77);

    let x = synth_x(in_f);
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let y_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();

    let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    rec.linear_native_rm(
        dtype,
        w_buf.as_ref(),
        stride_elems,
        x_buf.as_ref(),
        y_buf.as_ref(),
        in_f,
        out_f,
        2,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; out_f * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let resident = bits(&out);
    assert!(
        resident.iter().any(|&b| b != 0),
        "resident w_base output is all zeros"
    );

    let off = nonzero_off(dtype);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);
    let (arena, addr) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena.as_ref(), &backing).unwrap();
    let rec = be.recorder().unwrap();
    rec.linear_native_rm_streamed(
        dtype,
        addr + off as u64,
        stride_elems,
        x_buf.as_ref(),
        y_buf.as_ref(),
        in_f,
        out_f,
        2,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; out_f * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let streamed = bits(&out);

    for (i, (&r, &s)) in resident.iter().zip(streamed.iter()).enumerate() {
        assert_eq!(
            r,
            s,
            "w_base+arena_off: streamed differs from resident at out {i}: {} vs {}",
            f32::from_bits(r),
            f32::from_bits(s)
        );
    }
    println!("ok: w_base composes with a non-zero arena offset");
}

// ─── int8 dp4a (MMV) family — slice A2 ─────────────────────────────────────────────────────────
// Same resident-vs-streamed bitwise contract as the dequant GEMVs above, with one extra input
// stage: the activation is pre-quantized ONCE by quant_q8 (qa/dact/sact) and both legs read the
// SAME quantized buffers, so any difference is the weight path, never the activation path.

/// Quantized-activation buffers for `rows` rows of `in_f` f32s, produced on-GPU by `quant_q8`.
struct QAct {
    qa: Box<Buf2>,
    dact: Box<Buf2>,
    sact: Box<Buf2>,
}
type Buf2 = dyn infr_core::backend::Buffer;

fn quantize_x(be: &VulkanBackend, rows: usize, in_f: usize) -> QAct {
    let x: Vec<f32> = (0..rows * in_f)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05 + ((i / in_f) as f32) * 0.01)
        .collect();
    let x_buf = be.alloc(rows * in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let nblk = in_f / 32;
    let qa = be.alloc(rows * in_f, BufferUsage::Activations).unwrap();
    let dact = be.alloc(rows * nblk * 2, BufferUsage::Activations).unwrap();
    let sact = be.alloc(rows * nblk * 2, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.quant_q8(
        x_buf.as_ref(),
        qa.as_ref(),
        dact.as_ref(),
        sact.as_ref(),
        rows,
        in_f,
    );
    rec.finish().unwrap();
    QAct { qa, dact, sact }
}

/// Resident + both streamed legs for an int8-dp4a kernel; returns the three output-bit vectors.
#[allow(clippy::too_many_arguments)]
fn run_mmv_case(
    be: &VulkanBackend,
    name: String,
    dtype: DType,
    in_f: usize,
    out_f: usize,
    out_elems: usize,
    q: &QAct,
    dispatch_res: &dyn Fn(&infr_vulkan::Recorder, &Buf2, &Buf2),
    dispatch_str: &dyn Fn(&infr_vulkan::Recorder, u64, &Buf2),
) -> Case {
    let w_bytes = weight_bytes_for(dtype, in_f * out_f);
    let w = synth_weight_bytes(w_bytes, in_f * 31 + out_f);
    let y_buf = be.alloc(out_elems * 4, BufferUsage::Activations).unwrap();

    let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_res(&rec, w_buf.as_ref(), y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let resident = bits(&out);

    let (arena0, addr0) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena0.as_ref(), &w).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr0, y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let streamed_at0 = bits(&out);

    let off = nonzero_off(dtype);
    let mut backing = synth_weight_bytes(off, 0xBAD);
    backing.extend_from_slice(&w);
    let (arena1, addr1) = be.alloc_arena_bda(backing.len()).unwrap();
    be.upload(arena1.as_ref(), &backing).unwrap();
    let rec = be.recorder().unwrap();
    dispatch_str(&rec, addr1 + off as u64, y_buf.as_ref());
    rec.finish().unwrap();
    let mut out = vec![0u8; out_elems * 4];
    be.download(y_buf.as_ref(), &mut out).unwrap();
    let streamed_atoff = bits(&out);

    let _ = q; // the qa/dact/sact buffers are shared by both legs via the dispatch closures
    Case {
        name,
        resident,
        streamed_at0,
        streamed_atoff,
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_streamed_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    let q = quantize_x(&be, 1, in_f);
    for dtype in [DType::Q4K, DType::Q6K, DType::Iq4Xs] {
        let c = run_mmv_case(
            &be,
            format!("mmv dtype={dtype:?}"),
            dtype,
            in_f,
            out_f,
            out_f,
            &q,
            &|rec, w, y| {
                rec.linear_mmv(
                    dtype,
                    w,
                    0,
                    q.qa.as_ref(),
                    q.dact.as_ref(),
                    q.sact.as_ref(),
                    y,
                    in_f,
                    out_f,
                )
            },
            &|rec, addr, y| {
                rec.linear_mmv_streamed(
                    dtype,
                    addr,
                    0,
                    q.qa.as_ref(),
                    q.dact.as_ref(),
                    q.sact.as_ref(),
                    y,
                    in_f,
                    out_f,
                )
            },
        );
        assert_case(&c);
        println!("ok: {}", c.name);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mrow_streamed_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // in_f=256 exercises the OUTS4 layout (in_f < 2048), in_f=2048 the base 2-output layout;
    // rows 1/4 hit the m4 tier, 8 the MR=8 tier, 12 the -DMRV=16 tier. Q3_K covers the
    // funnel-shift (2-aligned block) load path.
    for in_f in [256usize, 2048] {
        let out_f = 8usize;
        for dtype in [DType::Q4K, DType::Q6K, DType::Q3K, DType::Q8_0] {
            for rows in [1usize, 4, 8, 12] {
                let q = quantize_x(&be, rows, in_f);
                let c = run_mmv_case(
                    &be,
                    format!("mmv_mrow dtype={dtype:?} rows={rows} in_f={in_f}"),
                    dtype,
                    in_f,
                    out_f,
                    rows * out_f,
                    &q,
                    &|rec, w, y| {
                        rec.linear_mmv_mrow(
                            dtype,
                            w,
                            0,
                            q.qa.as_ref(),
                            q.dact.as_ref(),
                            q.sact.as_ref(),
                            None,
                            y,
                            rows,
                            in_f,
                            out_f,
                        )
                    },
                    &|rec, addr, y| {
                        rec.linear_mmv_mrow_streamed(
                            dtype,
                            addr,
                            0,
                            q.qa.as_ref(),
                            q.dact.as_ref(),
                            q.sact.as_ref(),
                            y,
                            rows,
                            in_f,
                            out_f,
                        )
                    },
                );
                assert_case(&c);
                println!("ok: {}", c.name);
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mw_streamed_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f) = (256usize, 8usize);
    let q = quantize_x(&be, 1, in_f);
    for dtype in [DType::Q4K, DType::Q6K, DType::Q2K, DType::Q3K, DType::Q5K] {
        for warps in [4u32, 8] {
            let c = run_mmv_case(
                &be,
                format!("mmv_mw dtype={dtype:?} warps={warps}"),
                dtype,
                in_f,
                out_f,
                out_f,
                &q,
                &|rec, w, y| {
                    rec.linear_mmv_mw(
                        dtype,
                        warps,
                        w,
                        0,
                        q.qa.as_ref(),
                        q.dact.as_ref(),
                        q.sact.as_ref(),
                        None,
                        y,
                        in_f,
                        out_f,
                    )
                },
                &|rec, addr, y| {
                    rec.linear_mmv_mw_streamed(
                        dtype,
                        warps,
                        addr,
                        0,
                        q.qa.as_ref(),
                        q.dact.as_ref(),
                        q.sact.as_ref(),
                        y,
                        in_f,
                        out_f,
                    )
                },
            );
            assert_case(&c);
            println!("ok: {}", c.name);
        }
    }
}

// ─── tiled dp4a expert GEMM (MMQ) family — slice A3 ────────────────────────────────────────────
// Dense (non-EXPERT_GRID) parity per dtype. The `_xp*` EXPERT_GRID streamed twins are covered by
// composition: streamed-xp differs from resident-xp ONLY in the rb() weight source (proven here
// per dtype) and from streamed-dense ONLY in the EXPERT_GRID row/stride logic (proven by the
// existing resident `_xp` adapter tests — that logic is the SHARED #else arm of both compiles).
// The xp streamed dispatch lands with the expert router's `arena: Option<u64>` arm at integration.

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmq_dense_streamed_matches_resident() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // n%64, k%32 (and k%256 so every K-quant super-block is whole); m=8 pads the C tile to 64 rows.
    let (m, k, n) = (8usize, 256usize, 64usize);
    let q = quantize_x(&be, m, k);
    let c_rows = m.div_ceil(64) * 64;
    for dtype in [
        DType::Q4K,
        DType::Q6K,
        DType::Q8_0,
        DType::Q5_0,
        DType::Q5K,
        DType::Q5_1,
        DType::Q2K,
        DType::Q3K,
        DType::Q4_0,
        DType::Q4_1,
        DType::Iq4Nl,
        DType::Iq4Xs,
        DType::Iq2S,
        DType::Iq3S,
        DType::Q2_0,
    ] {
        let c = run_mmv_case(
            &be,
            format!("mmq dense dtype={dtype:?}"),
            dtype,
            k,
            n,
            c_rows * n,
            &q,
            &|rec, w, y| {
                rec.matmul_mmq(
                    dtype,
                    q.qa.as_ref(),
                    q.dact.as_ref(),
                    q.sact.as_ref(),
                    w,
                    0,
                    y,
                    m,
                    k,
                    n,
                )
            },
            &|rec, addr, y| {
                rec.matmul_mmq_streamed(
                    dtype,
                    q.qa.as_ref(),
                    q.dact.as_ref(),
                    q.sact.as_ref(),
                    addr,
                    0,
                    y,
                    m,
                    k,
                    n,
                )
            },
        );
        // Padded C rows (m..c_rows) hold garbage from staged-garbage activations in BOTH legs, and
        // the two legs' garbage can differ (the pad rows read whatever followed qa). Compare only
        // the real m rows.
        let real = m * n;
        let trimmed = Case {
            name: c.name,
            resident: c.resident[..real].to_vec(),
            streamed_at0: c.streamed_at0[..real].to_vec(),
            streamed_atoff: c.streamed_atoff[..real].to_vec(),
        };
        assert_case(&trimmed);
        println!("ok: {}", trimmed.name);
    }
}
