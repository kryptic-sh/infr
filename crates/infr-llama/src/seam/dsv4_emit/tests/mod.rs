//! The compressed tiers' tests: each compressor's ops alone against a host computation of its
//! definition (`compressor`), the synthetic V4 file through the runner's sessions (`session`),
//! and its batched prefill against one position per graph (`prefill`). llama.cpp scores the whole
//! model in `tests/synthetic_deepseek4_llama_cpp.rs`.

mod compressor;
mod paging;
mod prefill;
mod session;

use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_cpu::CpuBackend;
use infr_testkit::gguf::splitmix64;

/// `n` values uniform in `[-scale, scale)`, drawn from `seed`.
fn rand_vals(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = splitmix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            ((h >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn upload<T: bytemuck::Pod>(be: &CpuBackend, v: &[T], usage: BufferUsage) -> Box<dyn Buffer> {
    let bytes: &[u8] = bytemuck::cast_slice(v);
    let b = be.alloc(bytes.len().max(4), usage).expect("alloc");
    be.upload(b.as_ref(), bytes).expect("upload");
    b
}

fn read_f32(be: &CpuBackend, b: &dyn Buffer) -> Vec<f32> {
    let mut v = vec![0f32; b.len_bytes() / 4];
    be.download(b, bytemuck::cast_slice_mut(&mut v))
        .expect("download");
    v
}

fn read_f16(be: &CpuBackend, b: &dyn Buffer) -> Vec<f32> {
    let mut v = vec![0u16; b.len_bytes() / 2];
    be.download(b, bytemuck::cast_slice_mut(&mut v))
        .expect("download");
    v.iter()
        .map(|&h| half::f16::from_bits(h).to_f32())
        .collect()
}

/// The largest absolute difference between two equally long vectors.
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}
