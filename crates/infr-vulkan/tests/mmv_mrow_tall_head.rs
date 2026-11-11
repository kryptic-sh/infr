//! A head tall enough that the 4-output int8 GEMV (`native_mmv_mrow.comp`, `-DOUTS4`) needs more
//! than `Recorder::MAX_GROUP_COUNT_X` workgroups, so `dispatch_wide` splits it into a 2-D grid
//! whose padding workgroups recover an `o0` past `out_f`.
//!
//! A padding workgroup writes nothing, so its weight reads cannot change `y` however the bytes past
//! the tensor are filled: the host-reference comparison guards the 2-D index recovery and the last
//! real group, and the over-read itself is only observable to GPU-assisted validation. The weight
//! tensor is sized above the arena's block floor so it gets a block of its own, and a padding
//! workgroup that decodes its row reads up to a tensor-length past that block's end, which GPU-AV
//! reports as an out-of-bounds buffer-device-address read (see `gpuav_support`). Without the layer
//! installed the test verifies only the numeric result, not the bound, and says so on stderr.
//!
//! Run: cargo test -p infr-vulkan --test mmv_mrow_tall_head -- --ignored --nocapture
mod gpuav_support;

use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

fn f16v(lo: u8, hi: u8) -> f32 {
    half::f16::from_le_bytes([lo, hi]).to_f32()
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mrow_o4_tall_head_matches_host_reference() {
    let log = gpuav_support::enable("tall-head");
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // The int8 GEMV is built on packed int8 dot products (`dotPacked4x8AccSatEXT`).
    if !be.capabilities().i8_dot {
        eprintln!("skip: the int8 mrow GEMV needs packed int8 dot");
        return;
    }
    // in_f < 2048 takes the -DOUTS4 build (4 outputs per workgroup), so this vocabulary-sized
    // head is one group past the 1-D limit — the Gemma 3 shape (262144 rows, narrow embedding).
    let (in_f, out_f, m) = (256usize, 262_144usize, 1usize);
    let dt = DType::Q8_0;
    let (_, blk) = infr_gguf::block_layout(dt);
    let nblk = in_f / 32;
    let mut src: Vec<u8> = (0..out_f * nblk * blk)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    for (bi, b) in src.chunks_exact_mut(blk).enumerate() {
        b[0..2].copy_from_slice(&half::f16::from_f32(0.25 + (bi % 11) as f32 * 0.05).to_le_bytes());
    }
    let w = be
        .alloc(src.len().next_multiple_of(4), BufferUsage::Weights)
        .unwrap();
    be.upload(w.as_ref(), &src).unwrap();

    let xs: Vec<f32> = (0..m * in_f)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.021 + ((i % 13) as f32) * 0.004)
        .collect();
    let x = be.alloc(m * in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
    let qa = be.alloc(m * in_f, BufferUsage::Activations).unwrap();
    let dact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
    let sact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
    let y = be.alloc(m * out_f * 4, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.quant_q8(
        x.as_ref(),
        qa.as_ref(),
        dact.as_ref(),
        sact.as_ref(),
        m,
        in_f,
    );
    rec.linear_mmv_mrow(
        dt,
        w.as_ref(),
        0,
        qa.as_ref(),
        dact.as_ref(),
        sact.as_ref(),
        None,
        y.as_ref(),
        m,
        in_f,
        out_f,
    );
    rec.finish().unwrap();

    let read = |b: &dyn infr_core::backend::Buffer, n: usize| -> Vec<u8> {
        let mut out = vec![0u8; n];
        be.download(b, &mut out).unwrap();
        out
    };
    let qa_h: Vec<i32> = read(qa.as_ref(), m * in_f)
        .iter()
        .map(|&b| b as i8 as i32)
        .collect();
    let da_h: Vec<f32> = read(dact.as_ref(), m * nblk * 2)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16v(c[0], c[1]))
        .collect();
    let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&read(y.as_ref(), m * out_f * 4)).to_vec();

    // Same RMS-scaled metric as `mmv_mrow_legacy_formats`, against a byte-at-a-time Q8_0 replay.
    let mut worst_abs = 0f64;
    let mut sumsq = 0f64;
    for (o, &g) in got.iter().enumerate() {
        let mut acc = 0f64;
        for s in 0..nblk {
            let bd = (o * nblk + s) * blk;
            let d = f16v(src[bd], src[bd + 1]);
            let mut dp = 0i64;
            for j in 0..32 {
                dp += (src[bd + 2 + j] as i8 as i64) * qa_h[s * 32 + j] as i64;
            }
            acc += d as f64 * da_h[s] as f64 * dp as f64;
        }
        worst_abs = infr_testkit::nan_max64(worst_abs, (g as f64 - acc).abs());
        sumsq += acc * acc;
    }
    let mr = worst_abs / (sumsq / out_f as f64).sqrt();
    println!("Q8_0 in{in_f} out{out_f} m{m}: worst |Δ| / out-RMS = {mr:.2e}");
    assert!(
        mr < 2e-3,
        "tall o4 head diverges from the host reference ({mr:.2e} of the output RMS)"
    );

    drop(be);
    gpuav_support::assert_clean(&log, "mmv_mrow_o4_tall_head_matches_host_reference");
}
