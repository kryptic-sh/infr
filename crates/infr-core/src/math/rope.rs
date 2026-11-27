//! Rotary position embedding (RoPE) angles, in one place for every backend that computes them
//! on the host, so all of them agree bit for bit.
//!
//! Pair `p` of a `rope_dim`-wide rotation turns by `pos * theta^(-2p/rope_dim)`, divided by a
//! per-pair divisor when the model has one (`rope_freqs` / YaRN's `freq_factors`). The
//! frequencies depend only on the op's shape, so a caller builds them once per op with
//! [`rope_freqs`] and takes each angle's sine and cosine with [`rope_sin_cos`].

use super::{powf, sin_cos};

/// The largest RoPE angle, in radians, a backend is held to: `2^24`, past which a binary32
/// position is no longer an exact integer, so `pos as f32` (here and on the GPU) stops naming the
/// position. The Vulkan rotary shaders take their limit from this constant (infr-vulkan's
/// build.rs writes it into `rope_limits.glsl` as `ROPE_SINCOS_MAX`), and the seam refuses a
/// context whose angles would pass it.
pub const ROPE_ANGLE_MAX: f32 = 16_777_216.0;

/// `theta^(-2p/rope_dim)` for every pair `p` in `0..rope_dim/2`, the exponent rounded to
/// binary32 as `-2.0 * p as f32 / rope_dim as f32` (the expression the call sites wrote).
pub fn rope_freqs(theta: f32, rope_dim: usize) -> Vec<f32> {
    (0..rope_dim / 2)
        .map(|p| powf(theta, -2.0 * p as f32 / rope_dim as f32))
        .collect()
}

/// `(sin, cos)` of pair `p`'s angle at position `pos`: `pos * freq`, then divided by `divisor`
/// when there is one, each step rounded to binary32.
#[inline]
pub fn rope_sin_cos(pos: f32, freq: f32, divisor: Option<f32>) -> (f32, f32) {
    let mut ang = pos * freq;
    if let Some(d) = divisor {
        ang /= d;
    }
    sin_cos(ang)
}
