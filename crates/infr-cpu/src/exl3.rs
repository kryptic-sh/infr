//! Scalar EXL3 trellis decoder (CPU oracle, Phase 0 quality gate).
//!
//! Ported from [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3)
//! (MIT license), commit pin **`bd1f519172406a9560bd5c90e61000691bb69a07`** (v1.6.6,
//! 2026-10-31). Source files:
//!
//! - `exllamav3/exllamav3_ext/quant/exl3_dq.cuh` — `dq` / `dq2` / `dq4` / `dq2x2` /
//!   `dq8` (+ `aligned_*` / `half` variants) bit-window extraction
//! - `exllamav3/exllamav3_ext/quant/codebook.cuh` — `decode_3inst` / `decode_3inst_2`
//!   codebook functions for cb0 (`3inst`), cb1 (`mcg`), cb2 (`mul1`)
//! - `exllamav3/exllamav3_ext/rocm/quant/ptx_portable.cuh` — the `exl3_lop3_6a` C
//!   fallback (`(x & 0x8fff8fff) ^ 0x3b603b60`), used here because CPUs have no `lop3`
//!
//! # Adaptations from the CUDA original
//!
//! - The GPU decoders are bit-window extractors feeding tensor-core MMA fragments
//!   (`FragB`, `half2` pairs). This port decodes **one 16-bit window to one `f32`
//!   weight value**; the dot product it would feed is out of scope for the Phase 0
//!   oracle (perplexity is scored through a full-model harness, not a GEMV).
//! - `__funnelshift_r` / `fshift` are emulated with a `u64` merge+shift —
//!   bit-identical to the CUDA funnel shift for shifts in `0..64`.
//! - Half arithmetic (`__hadd`, `__hfma`) uses the [`half`] crate's `f16`, which
//!   rounds to nearest-even exactly like the hardware instructions, so the port is
//!   bit-identical to decoding on the GPU (verified by the parity tests below).
//! - The window extractors for every bitrate (`dq`, `dq4`, `dq2x2`, `dq8` and the
//!   `aligned_1/2/4bit` + `half` specializations) all reduce to the same scalar
//!   operation — extract overlapping 16-bit windows from a `bits * 256`-bit tile —
//!   so one bit-reader serves all rates. Half-integer rates (`dq8_half`) are NOT
//!   ported: they are a tiling detail of the GPU kernels, and Phase 0 only needs
//!   the integer rates cb0 = 2 bpw, cb1 = 3 bpw, cb2 = 4 bpw.
//! - Full-tensor layout (16x16 tiles, Hadamard + `suh`/`svh` scales) is Phase 1
//!   scope. This module decodes **one 256-weight tile** given its trellis words,
//!   which is the unit the CUDA `dq*` family operates on. The second half of this
//!   module (below [`decode_matrix`]) lifts tiles to full `(k, n)` matrices:
//!   the in-tile permutation ([`tile_perm`], transliterated from
//!   `reconstruct_tile`'s shuffle/scatter/readback in `reconstruct.cu`), the
//!   `suh`/`svh` diagonal scales, and the orthonormal 128-wide Hadamard
//!   ([`hadamard128`], `r_scale = scale / sqrt(128)` per `hadamard.cu`).
//!
//! Run: `cargo test -p infr-cpu --lib exl3` (unit goldens) and
//! `cargo test -p infr-cpu --test exl3_parity` (seeded bit-parity sweep).
//!
//! # Phase-1 math (from `modules/quant/exl3.py` + `reconstruct.cu`)
//!
//! For a linear with trellis `T`, scales `suh` (k) / `svh` (n), bias `b`:
//!
//! ```text
//! D[k,n] = decode(T)                       // rotated-basis weight (this module)
//! S[k,n] = D[k,n] * suh[k] * svh[n]        // [`apply_scales`]
//! xh = H * (suh * x) / sqrt(128)           // input pre-scale + Hadamard
//! y_ = D^T * xh                            // GEMV/GEMM against the decoded trellis
//! y  = svh * (H * y_) / sqrt(128) + b      // output Hadamard + post-scale + bias
//! ```
//!
//! Equivalently `y = x @ W + b` with the folded weight
//! `W = diag(suh) * H * D * H * diag(svh) / 128` (the `reconstruct_had_tile`
//! comment in `reconstruct.cu`), which is what the prefill path materializes.
//! `H` is the symmetric orthonormal FWHT ([`hadamard128`]); `suh` scales the
//! input side, `svh` the output side (see `had_r_128(x, xh, suh, None, 1.0)`
//! and `had_r_128(y_, y_, None, svh, 1.0)` in `exl3.py::forward`).

use half::f16;

/// Trellis codebook, matching exllamav3's `codebook` names (`tests/testlib/exl3.py`:
/// `CODEBOOKS = ("3inst", "mcg", "mul1")`) and the `cb` template parameter of
/// `decode_3inst<cb>` in `codebook.cuh`.
///
/// Bitrate association (issue #6, Phase 0 plan): cb2 ≈ 4 bpw, cb1 ≈ 3 bpw,
/// cb0 ≈ 2 bpw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codebook {
    /// `3inst` — hash + add path (`x *= 89226354; x += 64248484`).
    Cb0,
    /// `mcg` — multiplicative congruential generator path (`x *= 0xCBAC1FED`).
    /// This is the **3 bpw** codebook the Phase 0 gate is decided on.
    Cb1,
    /// `mul1` — byte-sum path (`x *= 0x83DCD12D`, dp4a over the four bytes).
    /// This is the **4 bpw** codebook.
    Cb2,
}

/// `exl3_lop3_6a` from `ptx_portable.cuh` (the ROCm C fallback for the
/// `lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a` PTX instruction):
/// turns a 32-bit hash product into two f16 bit patterns.
#[inline]
pub fn lop3_6a(x: u32) -> u32 {
    (x & 0x8fff8fff) ^ 0x3b603b60
}

/// Byte sum `__dp4a(x, 0x01010101, acc)`: sum of the four bytes of `x` plus `acc`.
#[inline]
fn dp4a_sum(x: u32, acc: u32) -> u32 {
    acc + (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + ((x >> 24) & 0xff)
}

/// `decode_3inst<cb>(x)` from `codebook.cuh`: one 16-bit trellis word to one `f16`
/// weight value, returned here widened to `f32` (matches `decode_3inst_f`).
pub fn decode_3inst(cb: Codebook, w: u16) -> f32 {
    f16::from_bits(decode_3inst_h(cb, w)).to_f32()
}

/// `f16`-bit-exact variant: returns the raw `f16` bits, for tests that must pin
/// the hardware rounding (the `__hadd`/`__hfma` in the CUDA original round once,
/// in f16, before the widening conversion).
pub fn decode_3inst_h(cb: Codebook, w: u16) -> u16 {
    match cb {
        Codebook::Cb0 => {
            let x = (w as u32).wrapping_mul(89226354).wrapping_add(64248484);
            hadd_halves(lop3_6a(x))
        }
        Codebook::Cb1 => {
            let x = (w as u32).wrapping_mul(0xCBAC1FED);
            hadd_halves(lop3_6a(x))
        }
        Codebook::Cb2 => {
            const ACC: u32 = 0x6400;
            let x = (w as u32).wrapping_mul(0x83DCD12D);
            let sum = dp4a_sum(x, ACC);
            // `half_uint16 h((uint16_t) sum); __hfma(h, k_inv, k_bias)` — one
            // correctly-rounded f16 FMA (see `f16_fma`), then the caller widens.
            let h = f16::from_bits(sum as u16);
            let k_inv = f16::from_bits(0x1eee); // 1/147.7
            let k_bias = f16::from_bits(0xc931); // -10.39
            f16_fma(h, k_inv, k_bias).to_bits()
        }
    }
}

/// Correctly-rounded f16 fused multiply-add: `a * b + c` with a SINGLE
/// round-to-nearest-even to f16, matching CUDA `__hfma` (default RN mode).
///
/// WHY NOT `half`'s `mul_add`: `half` 2.x only provides `mul_add` through the
/// `num_traits::Float` impl, which widens to f32 and rounds TWICE (once in f32,
/// once back to f16). The `f64` detour here is exact, not approximate: every f16
/// value is exact in f64, the product of two 11-bit mantissas needs at most 22
/// bits, and adding the third operand (same order of magnitude here) stays far
/// under f64's 53-bit mantissa — so `from_f64` performs the single IEEE rounding
/// that `__hfma` does in hardware.
#[inline]
fn f16_fma(a: f16, b: f16, c: f16) -> f16 {
    f16::from_f64(a.to_f64() * b.to_f64() + c.to_f64())
}

/// Reinterpret the two f16 halves of `x` and return their correctly-rounded f16
/// sum (`__hadd(__low2half, __high2half)`), as raw bits.
///
/// `half`'s `f16` addition is a single RN-even rounding (f16 operands are exact
/// in f32 and their sum needs at most 13 significand bits, so the widen-add-
/// narrow path rounds exactly once), matching `__hadd`'s default RN mode.
#[inline]
fn hadd_halves(x: u32) -> u16 {
    let lo = f16::from_bits(x as u16);
    let hi = f16::from_bits((x >> 16) as u16);
    (lo + hi).to_bits()
}

/// `decode_3inst_2<cb>(w0, w1)` from `codebook.cuh`: the paired form the `dq2` /
/// `dq4` / `dq8` extractors use. Each lane is independent — the pairing is pure
/// instruction-level parallelism on the GPU — so this is exactly two `decode_3inst`
/// calls, kept as one function so the tile decoder mirrors the CUDA structure.
pub fn decode_3inst_2(cb: Codebook, w0: u16, w1: u16) -> (f32, f32) {
    (decode_3inst(cb, w0), decode_3inst(cb, w1))
}

/// CUDA `fshift` / `__funnelshift_r(b, a, s)` semantics from `exl3_dq.cuh`:
/// `concat(a, b) >> s`, i.e. `a` is the HIGH word, `b` the LOW word.
///
/// `__funnelshift_r(lo, hi, shift)` returns `(hi << (32 - shift)) | (lo >> shift)`
/// for `shift < 32` (and `hi >> (shift - 32)` above); the `u64` merge+shift in
/// `fshift` is bit-identical for all shifts and is what the generic `dq4`/`dq8`
/// paths use, so it is the canonical form here.
#[inline]
fn funnel(a: u32, b: u32, shift: u32) -> u32 {
    ((((a as u64) << 32) | (b as u64)) >> shift) as u32
}

/// Extract the 16-bit trellis window at absolute tile bit position `bit`
/// (funnel-shift path — mirrors the CUDA `dq*` extractors: load the two bracketing
/// u32 words, funnel-shift into place, mask to 16 bits).
///
/// BIT ORDER (verified against the CUDA indexing: window `[b, b+16)` of word
/// `0x12345678` at `b = 0` yields `0x1234`, at `b = 16` yields `0x5678`): stream
/// bit `k` is word bit `31 - (k % 32)` — big-endian within each word — and the
/// window's FIRST stream bit is its MSB. The funnel shift implements this
/// directly, so this function is an exact transcription of the CUDA form.
///
/// `words.len()` must be `bits * 256 / 32` (the tile size); indices wrap modulo
/// the tile exactly like the CUDA `ptr[i % (bits * 256 / 32)]`.
pub fn window16_funnel(words: &[u32], bit: usize) -> u16 {
    let n = words.len();
    debug_assert!(n > 0);
    let b1 = bit + 16; // end of window
    let i0 = bit / 32; // word containing the first bit
    let i2 = (b1 - 1) / 32; // word containing the last bit (may == i0)
    let s2 = ((i2 + 1) * 32 - b1) as u32; // shift to align the window low
    let a = words[i0 % n];
    let b = words[i2 % n];
    (funnel(a, b, s2) & 0xffff) as u16
}

/// The same window read by direct bit-slicing — the INDEPENDENT oracle path for
/// the parity tests. Treats the tile as a bit array with the CUDA stream order
/// (stream bit `k` = word bit `31 - (k % 32)`, window MSB first) and gathers 16
/// bits one by one; shares no arithmetic with [`window16_funnel`], so agreement
/// proves the funnel math, not a shared typo.
pub fn window16_slice(words: &[u32], bit: usize) -> u16 {
    let n = words.len();
    let mut w = 0u16;
    for k in 0..16 {
        let kk = bit + k;
        let word = words[(kk / 32) % n];
        w |= (((word >> (31 - (kk % 32))) & 1) as u16) << (15 - k);
    }
    w
}

/// Decode one 256-weight tile at `bits` bits/weight with codebook `cb`,
/// in `t_offset` order `0..256`.
///
/// This is the scalar form of the CUDA `dq<bits, cb>` extractor
/// (`exl3_dq.cuh`):
///
/// ```text
/// b0 = t * bits + bits - 16 + 256 * bits   (start of this weight's window)
/// i0 = b0 / 32, i1 = (b0 + 15) / 32, s0 = (i1 + 1) * 32 - (b0 + 16)
/// w0 = funnelshift_r(ptr[i1], ptr[i0], s0) & 0xffff
/// out[t] = decode_3inst<cb>(w0)
/// ```
///
/// `tile_u32` holds the tile's `bits * 256 / 32` u32 words — the same word
/// array the CUDA kernel reads as `ptr` (u32 view over the tile bytes); the returned `Vec` has 256 `f32`
/// values, one per weight, before scales/Hadamard (Phase 1 scope).
///
/// `bits` must be 1..=8 (integer rates only; half-integer `dq8_half` tiling is a
/// GPU-kernel detail with no Phase 0 consumer).
pub fn decode_tile(cb: Codebook, bits: u32, tile_u32: &[u32]) -> Vec<f32> {
    assert!(
        (1..=8).contains(&bits),
        "EXL3 Phase-0 oracle supports integer bits 1..=8, got {bits}"
    );
    assert_eq!(
        tile_u32.len(),
        (bits as usize) * 256 / 32,
        "tile must hold bits*256/32 u32 words"
    );
    let base = 256 * (bits as usize);
    let mut out = Vec::with_capacity(256);
    for t in 0..256 {
        // `dq`: b0 = t*bits + bits - 16 + 256*bits, window = [b0, b0+16).
        // Signed: the CUDA `b0` is a signed int (the `- 16` underflows usize
        // for small t*bits+bits, e.g. bits=1, t=0 -> 1-16+256 = 241).
        let b0 = (t * (bits as usize) + (bits as usize) + base) as isize - 16;
        debug_assert!(b0 >= 0);
        out.push(decode_3inst(cb, window16_funnel(tile_u32, b0 as usize)));
    }
    out
}

/// Tile decode through the independent slice oracle — same weights, none of the
/// funnel arithmetic. `exl3_parity` asserts this matches [`decode_tile`] bit-for-bit.
pub fn decode_tile_oracle(cb: Codebook, bits: u32, tile_u32: &[u32]) -> Vec<f32> {
    assert!((1..=8).contains(&bits));
    assert_eq!(tile_u32.len(), (bits as usize) * 256 / 32);
    let base = 256 * (bits as usize);
    let mut out = Vec::with_capacity(256);
    for t in 0..256 {
        let b0 = (t * (bits as usize) + (bits as usize) + base) as isize - 16;
        debug_assert!(b0 >= 0);
        out.push(decode_3inst(cb, window16_slice(tile_u32, b0 as usize)));
    }
    out
}

/// Number of u16 trellis words in one (16, 16) tile at `K` bits/weight, per the
/// checkpoint layout (`tests/testlib/exl3.py`: `trellis (k//16, n//16, 16*K)`).
/// Integer rates only (Phase 0 scope).
pub fn tile_words_u16(bits: u32) -> usize {
    assert!((1..=8).contains(&bits));
    16 * (bits as usize)
}

// ─── Phase 1: full-tensor layout ─────────────────────────────────────────────
// Tiles to matrices: the in-tile permutation, the suh/svh scales, the Hadamard.
//
// Reference: `reconstruct_tile` in `reconstruct.cu` (decode + tensor-core shuffle
// + scatter to row-major) and `had_hf/ff_r_128_inner` in `hadamard_inner.cuh`.

/// In-tile permutation: stream index `s` (0..256, the `t_offset` order of [`decode_tile`])
/// maps to within-tile `(row, col)` = `(perm[s] / 16, perm[s] % 16)`.
///
/// Transliterated from `reconstruct_tile` (`reconstruct.cu`): each warp decodes one
/// 16x16 tile (`dq_dispatch(ptr, lane * 8, ...)` — lane `L` owns stream words
/// `8*L..8*L+8`), the `__shfl_down_sync(..., 4)` + `tile[r][warp][c]` scatter routes
/// `(lane, frag_entry)` to tile coordinates, and the `tile_int4[t]` readback copies
/// tile halves to row-major `(k*16 + t/16, group*128 + (t%16)*8 + l)`.
///
/// Readback subtlety (`tile_int4[t] = ...`, `*out_int4 = tile_int4[t]`): an int4
/// is FOUR half2, so readback quad `t` carries tile half2 `{4t..4t+3}` — tile
/// row `r = t/16` (matching the output row), tile-in-group `W = (t%16)/2`, and
/// tile col-pairs `c2 = 4*(t%2) + {0..3}`. Within-tile column is therefore
/// `8*(c2 >= 4) + 2*(c2%4) + hi`, NOT `2*c2 + hi` (that misreading failed
/// `tile_perm_is_permutation`; the test earns its keep).
///
/// The routing (shuffle/scatter/readback) is rate-independent; only the
/// `(lane, entry) -> stream word` step depends on `bits` (see `stream_index`).
/// `tile_perm_is_permutation` asserts the composed map is a bijection (every
/// weight lands in exactly one matrix slot — a dropped or doubled weight would
/// silently corrupt the model), and `fragment_entries_match_scalar_words`
/// pins the rate-specific step against an independent bit-op transliteration.
pub fn tile_perm(bits: u32) -> [u16; 256] {
    assert!((1..=8).contains(&bits));
    let mut perm = [0u16; 256];
    // Readback half2 coordinates (r_t, w_t, c2_t, hi): tile[r_t][w_t][c2_t] low/high half.
    // Inverting the scatter: the storing lane and frag entry behind each tile half.
    for r_t in 0..16usize {
        // Storing-lane residue: r_t = 2*(L%4) + {0,1} (+8 for the second row pair).
        let m = (r_t % 8) / 2;
        let jr = (r_t % 2) + if r_t >= 8 { 2 } else { 0 };
        for c2 in 0..8usize {
            // Storing lane: c2 in 0..4 -> c0 = L/8; c2 in 4..8 -> c1 = L/8 + 4.
            let l = 8 * (c2 % 4) + m;
            let e = jr + if c2 >= 4 { 4 } else { 0 };
            // Within-tile column from the readback interleave (see above).
            let c16 = 8 * usize::from(c2 >= 4) + 2 * (c2 % 4);
            for hi in 0..2usize {
                let s = stream_index(bits, (l + hi * 4) as u32, e as u32);
                perm[s as usize] = (r_t * 16 + c16 + hi) as u16;
            }
        }
    }
    perm
}

/// Scalar stream word behind fragment entry `e` of warp lane `lane`.
///
/// `reconstruct_tile` calls `dq_dispatch(s_packed[warp], lane * 8, frag0, frag1)`.
/// Every rate variant decodes 8 consecutive scalar words `8*lane..8*lane+8`
/// (entries `e0..e7 = w0..w7`): `dq8`/`dq4`/`dq2x2` by the window arithmetic in
/// `exl3_dq.cuh` (`b0 = (t_offset+257)*bits - 16`, windows spaced `bits` apart);
/// the `aligned_1/2/4bit` specializations by the independent transliteration
/// checked in `fragment_entries_match_scalar_words`.
#[inline]
fn stream_index(_bits: u32, lane: u32, entry: u32) -> u32 {
    debug_assert!(entry < 8);
    lane * 8 + entry
}

/// Decode a full `(k, n)` weight matrix from checkpoint trellis words (rotated basis,
/// UNSCALED — caller applies [`apply_scales`]).
///
/// `trellis_u16` is the raw `*.trellis` tensor contents (`(k//16, n//16, 16*K)`
/// int16, row-major): tile `(ki, nj)` holds `16*K` u16 words. Output is `(k, n)`
/// row-major, one `f32` per weight, tile interiors ordered by [`tile_perm`].
pub fn decode_matrix(cb: Codebook, bits: u32, trellis_u16: &[u16], k: usize, n: usize) -> Vec<f32> {
    assert!((1..=8).contains(&bits));
    assert!(k.is_multiple_of(16) && n.is_multiple_of(16));
    let w = tile_words_u16(bits);
    let (kti, nt) = (k / 16, n / 16);
    assert_eq!(trellis_u16.len(), kti * nt * w);
    let perm = tile_perm(bits);
    let mut out = vec![0f32; k * n];
    let mut tile32 = vec![0u32; w / 2];
    for ki in 0..kti {
        for nj in 0..nt {
            let base = (ki * nt + nj) * w;
            for (i, pair) in trellis_u16[base..base + w]
                .as_chunks::<2>()
                .0
                .iter()
                .enumerate()
            {
                tile32[i] = (pair[0] as u32) | ((pair[1] as u32) << 16);
            }
            let vals = decode_tile(cb, bits, &tile32);
            for (s, v) in vals.iter().enumerate() {
                let p = perm[s] as usize;
                out[(ki * 16 + p / 16) * n + nj * 16 + p % 16] = *v;
            }
        }
    }
    out
}

/// Fold the diagonal scales: `mat[r*n+c] *= suh[r] * svh[c]` in place.
///
/// Upstream applies them around the Hadamards (`had_r_128(x, xh, suh, ...)` pre,
/// `had_r_128(y_, y_, None, svh, ...)` post); the folded-matrix view in
/// `reconstruct_columns` is the same product (`w *= suh` rows, `w *= svh` cols).
/// `suh`/`svh` arrive as f16 in the checkpoint — convert to f32 on load.
pub fn apply_scales(mat: &mut [f32], suh: &[f32], svh: &[f32], k: usize, n: usize) {
    assert_eq!(mat.len(), k * n);
    assert_eq!(suh.len(), k);
    assert_eq!(svh.len(), n);
    for (r, row) in mat.chunks_exact_mut(n).enumerate() {
        let a = suh[r];
        for (c, v) in row.iter_mut().enumerate() {
            *v *= a * svh[c];
        }
    }
}

/// In-place orthonormal 128-point Hadamard (FWHT + `/sqrt(128)`).
///
/// Matches `had_hf/ff_r_128_inner` with `r_scale = scale / sqrt(128)` (`hadamard.cu`;
/// `0.088388347648f`): pre-scale, butterfly, single `/sqrt(128)` normalization.
/// `H` is symmetric and orthonormal: `H(H(x)) == x`, `|H(x)| == |x|`.
pub fn hadamard128(v: &mut [f32]) {
    assert_eq!(v.len(), 128);
    let mut len = 1usize;
    while len < 128 {
        for i in (0..128).step_by(2 * len) {
            for j in 0..len {
                let (a, b) = (v[i + j], v[i + j + len]);
                v[i + j] = a + b;
                v[i + j + len] = a - b;
            }
        }
        len *= 2;
    }
    // 1/sqrt(128): the exact CUDA constant, not a recomputation.
    const R_SCALE: f32 = 0.088388347648_f64 as f32;
    for x in v.iter_mut() {
        *x *= R_SCALE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `lop3_6a(0) == 0x3b603b60`: the AND kills everything, the XOR stamps the
    /// constant — pins the two magic constants against the ROCm fallback.
    /// (And `lop3_6a(!0) == 0x8fff8fff ^ 0x3b603b60`: the AND is the identity.)
    #[test]
    fn lop3_identity_constant() {
        assert_eq!(lop3_6a(0), 0x3b603b60);
        assert_eq!(lop3_6a(0xffff_ffff), 0x8fff8fff ^ 0x3b603b60);
    }

    /// `hadd_halves(0x3b603b60)`: both halves are f16 `0x3b60` = 0.921875, so the
    /// correctly-rounded f16 sum is exactly 1.84375 (`0x3db6`).
    #[test]
    fn hadd_halves_golden() {
        let h = hadd_halves(0x3b603b60);
        assert_eq!(h, f16::from_f32(1.84375).to_bits());
        assert_eq!(f16::from_bits(h).to_f32(), 1.84375);
    }

    /// cb2 zero-word golden, recomputed here in f64 from the definition (not via
    /// `decode_3inst`, so this is an oracle, not a tautology):
    /// `x = 0 * 0x83DCD12D = 0`, `sum = 0 + 0x6400 = 25600`, `h = f16(0x6400)`,
    /// `out = h * (1/147.7) - 10.39` in f16 FMA arithmetic.
    #[test]
    fn cb2_zero_word_matches_f64_oracle() {
        let got = decode_3inst(Codebook::Cb2, 0);
        assert!(got.is_finite());
        // f16(0x6400): sign 0, exp 25, mantissa 0 -> 1.0 * 2^(25-15) = 1024.0.
        assert_eq!(f16::from_bits(0x6400).to_f32(), 1024.0);
        // The f16 FMA rounds once; f64 recomputation must land within 1 ulp of
        // the f16 result (≈ 2^-11 relative at this magnitude).
        let approx = 1024.0f64 * (1.0 / 147.7) - 10.39;
        let rel = ((got as f64 - approx) / approx).abs();
        assert!(rel < 2e-3, "got {got}, f64 oracle {approx}");
        // Full determinism pin: the exact f32 of the f16-rounded result.
        let h = f16::from_bits(0x6400);
        let exact = f16_fma(h, f16::from_bits(0x1eee), f16::from_bits(0xc931));
        assert_eq!(decode_3inst_h(Codebook::Cb2, 0), exact.to_bits());
    }

    /// The paired decoder is exactly two scalar decodes (the pairing is GPU
    /// instruction parallelism, not math).
    #[test]
    fn pair_is_two_scalars() {
        for cb in [Codebook::Cb0, Codebook::Cb1, Codebook::Cb2] {
            for w in [0u16, 1, 0x1234, 0x8000, 0xffff] {
                let (a, b) = decode_3inst_2(cb, w, w.wrapping_add(7));
                assert_eq!(a, decode_3inst(cb, w));
                assert_eq!(b, decode_3inst(cb, w.wrapping_add(7)));
            }
        }
    }

    /// Funnel and slice windows agree on every alignment, including the
    /// single-word (`i0 == i2`) and wrap-around (`% tile_len`) cases.
    #[test]
    fn funnel_matches_slice_on_all_alignments() {
        // 4-bit tile: 32 words. PRBS-ish contents, no zero words.
        let words: Vec<u32> = (0..32)
            .map(|i| (i as u32).wrapping_mul(2654435761).wrapping_add(0x9e3779b9))
            .collect();
        for bit in 0..32 * 32 {
            assert_eq!(
                window16_funnel(&words, bit),
                window16_slice(&words, bit),
                "window at bit {bit}"
            );
        }
    }

    /// Tile decode is deterministic and finite over adversarial bytes at every
    /// integer rate x every codebook.
    #[test]
    fn tile_decode_finite_everywhere() {
        for bits in 1..=8u32 {
            let n = bits as usize * 8;
            let words: Vec<u32> = (0..n)
                .map(|i| {
                    (i as u32)
                        .wrapping_mul(0x85ebca6b)
                        .wrapping_add(bits.wrapping_mul(0x27d4eb2f))
                })
                .collect();
            for cb in [Codebook::Cb0, Codebook::Cb1, Codebook::Cb2] {
                let out = decode_tile(cb, bits, &words);
                assert_eq!(out.len(), 256);
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "non-finite at bits={bits} cb={cb:?}"
                );
            }
        }
    }

    // ── Phase-1 layout tests ────────────────────────────────────────────

    /// Deterministic PRBS for the layout tests (xorshift64, no `rand` dep).
    fn xrs(state: &mut u64) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 32) as u32
    }

    fn prbs_tile(bits: u32, seed: u64) -> Vec<u32> {
        let n = bits as usize * 8;
        let mut s = seed | 1;
        (0..n).map(|_| xrs(&mut s)).collect()
    }

    /// Scalar window start of stream word `s` (the `dq` formula), for pinning
    /// the fragment extractors below against [`window16_funnel`].
    fn scalar_b0(bits: u32, s: u32) -> usize {
        (s as usize) * (bits as usize) + (bits as usize) + 256 * (bits as usize) - 16
    }

    /// Exact transliteration of the `aligned_*` + `dq8`/`dq4`/`dq2x2` fragment
    /// extractors' EIGHT windows for the lane decoded at `idx` (`dq_dispatch`
    /// is always called with `lane * 8`). Entry order `e0..e7 = w0..w7`
    /// (`frag0[0] = (w0,w1)`, `frag0[1] = (w2,w3)`, `frag1[0] = (w4,w5)`,
    /// `frag1[1] = (w6,w7)`), matching [`stream_index`]'s contract.
    ///
    /// Bit-op details from `exl3_dq.cuh`: `FSHF_IMM(s, b, a, imm)` is read as
    /// `funnelshift_r(b, a, imm)` (same arg order as the scalar `dq`'s
    /// `__funnelshift_r(b, a, s0)`); `BFE16_IMM(w, b, n)` is `(b >> n) & 0xffff`.
    /// If either macro reads the other way, this test fails and the shift
    /// direction gets revisited — it does NOT silently adapt.
    fn fragment_windows(bits: u32, words: &[u32], idx: u32) -> [u16; 8] {
        let nwords = words.len() as u32;
        let at = |i: u32| words[(i % nwords) as usize];
        let mut w = [0u16; 8];
        match bits {
            1 => {
                let i1 = idx >> 5;
                let a = at(i1.wrapping_add(nwords - 1) % nwords);
                let b0 = at(i1);
                let b = funnel(a, b0, (!idx) & 24);
                for j in 0..8 {
                    w[j] = ((b >> (7 - j as u32)) & 0xffff) as u16;
                }
            }
            2 => {
                let i1 = idx >> 4;
                let a = at(i1.wrapping_add(nwords - 1) % nwords);
                let b0 = at(i1);
                let b = funnel(a, b0, ((!idx) & 8) << 1);
                for j in 0..8 {
                    w[j] = ((b >> (2 * (7 - j as u32))) & 0xffff) as u16;
                }
            }
            3 => {
                // dq8<bits=3, align=4>: two funnel shifts + shift chains.
                // NOTE the shifts operate on UNMASKED 32-bit values (masks
                // apply at decode); the transliteration keeps full u32s too.
                let b1 = (idx + 257) * 3;
                let b2 = b1 + 3 * 7;
                let b0 = b1 - 16;
                let i0 = b0 / 32;
                let i2 = (b2 - 1) / 32;
                let s2 = (i2 + 1) * 32 - b2;
                let a = at(i0);
                let b = at(i2);
                let mut f = [0u32; 8];
                f[7] = funnel(a, b, s2);
                f[6] = f[7] >> 3;
                f[5] = f[6] >> 3;
                f[4] = f[5] >> 3;
                f[3] = funnel(a, b, s2 + 3 * 4);
                f[2] = f[3] >> 3;
                f[1] = f[2] >> 3;
                f[0] = f[1] >> 3;
                for j in 0..8 {
                    w[j] = (f[j] & 0xffff) as u16;
                }
            }
            4 => {
                let i1 = idx >> 3;
                let a = at(i1.wrapping_add(nwords - 1) % nwords);
                let b0 = at(i1);
                let s = funnel(a, b0, 20);
                w[7] = (b0 & 0xffff) as u16;
                w[6] = ((b0 >> 4) & 0xffff) as u16;
                w[5] = ((b0 >> 8) & 0xffff) as u16;
                w[4] = ((b0 >> 12) & 0xffff) as u16;
                w[3] = ((b0 >> 16) & 0xffff) as u16;
                w[2] = (s & 0xffff) as u16;
                w[1] = ((s >> 4) & 0xffff) as u16;
                w[0] = ((s >> 8) & 0xffff) as u16;
            }
            5 | 6 | 8 => {
                // dq4(idx) -> e0..e3, dq4(idx+4) -> e4..e7: the
                // (t_offset+257)*bits - 16 window arithmetic, textually
                // independent of the scalar dq's t*bits + bits - 16 + 256*bits.
                for (g, base) in [(0, idx), (4, idx + 4)] {
                    let b0 = (base + 257) * bits - 16;
                    let b2 = b0 + 3 * bits + 16;
                    let i0 = b0 / 32;
                    let i2 = (b2 - 1) / 32;
                    let s2 = (i2 + 1) * 32 - b2;
                    let a = at(i0);
                    let b = at(i2);
                    w[g] = (funnel(a, b, s2 + 3 * bits) & 0xffff) as u16;
                    w[g + 1] = (funnel(a, b, s2 + 2 * bits) & 0xffff) as u16;
                    w[g + 2] = (funnel(a, b, s2 + bits) & 0xffff) as u16;
                    w[g + 3] = (funnel(a, b, s2) & 0xffff) as u16;
                }
            }
            7 => {
                // dq2x2(idx) -> e0..e3, dq2x2(idx+4) -> e4..e7.
                for (g, base) in [(0, idx), (4, idx + 4)] {
                    for i in 0..2u32 {
                        let b0 = (base + 2 * i + 257) * bits - 16;
                        let b2 = b0 + bits + 16;
                        let i0 = b0 / 32;
                        let i2 = (b2 - 1) / 32;
                        let s2 = (i2 + 1) * 32 - b2;
                        let a = at(i0);
                        let b = at(i2);
                        w[g + (2 * i) as usize] = (funnel(a, b, s2 + bits) & 0xffff) as u16;
                        w[g + (2 * i + 1) as usize] = (funnel(a, b, s2) & 0xffff) as u16;
                    }
                }
            }
            _ => panic!("integer rates 1..=8, got {bits}"),
        }
        w
    }

    /// Every fragment entry of every lane must be the scalar window of the
    /// matching stream word (`idx + entry`): the rate-specific half of the
    /// [`tile_perm`] composition. For `dq8`/`dq4`/`dq2x2` this re-derives the
    /// window arithmetic through a second code path; for the `aligned_*`
    /// specializations the bit ops are genuinely independent of the scalar
    /// funnel reader, so agreement pins both (including the `FSHF_IMM` shift
    /// direction and the MSB-first stream order).
    #[test]
    fn fragment_entries_match_scalar_words() {
        for bits in 1..=8u32 {
            let words = prbs_tile(bits, 0x1234_5678_9abc_def0 ^ (bits as u64));
            for lane in 0..32u32 {
                let idx = lane * 8;
                let got = fragment_windows(bits, &words, idx);
                for (j, g) in got.iter().enumerate() {
                    let want = window16_funnel(&words, scalar_b0(bits, idx + j as u32));
                    assert_eq!(
                        *g, want,
                        "bits={bits} lane={lane} entry={j}: fragment {g:#06x} != scalar {want:#06x}"
                    );
                }
            }
        }
    }

    /// The composed in-tile map must be a bijection: 256 stream words to 256
    /// distinct `(row, col)` slots. A collision would stack two weights in one
    /// slot (and leave another zero); a gap would drop a weight — both
    /// silently corrupt the matrix.
    #[test]
    fn tile_perm_is_permutation() {
        for bits in 1..=8u32 {
            let mut seen = tile_perm(bits);
            seen.sort_unstable();
            for (i, v) in seen.iter().enumerate() {
                assert_eq!(*v as usize, i, "bits={bits}: perm is not bijective at {i}");
            }
        }
    }

    /// `pack_trellis_kernel` transliteration (`pack.cu`): 256 K-bit codes to
    /// `16*K` packed u16 words, 16 spans of 16 weights MSB-first, with the
    /// kernel's `SWAP16` on the u32 store.
    fn pack_codes(codes: &[u16], bits: u32) -> Vec<u16> {
        assert_eq!(codes.len(), 256);
        let k = bits;
        let mut packed = vec![0u16; 16 * k as usize];
        for span in 0..16usize {
            let (mut i, mut j, mut kk, mut buf) = (16 * span, k as usize * span, 32u32, 0u32);
            for _ in 0..16 {
                let v = (codes[i] as u32) & ((1 << k) - 1);
                kk -= k;
                buf |= v << kk;
                if kk <= 16 {
                    packed[j] = (buf >> 16) as u16;
                    buf <<= 16;
                    kk += 16;
                    j += 1;
                }
                i += 1;
            }
        }
        // SWAP16 on each u32: swap the two u16 halves.
        for pair in packed.as_chunks_mut::<2>().0.iter_mut() {
            pair.swap(0, 1);
        }
        packed
    }

    /// Reference bitstream packer: 256 K-bit codes to u32 words, MSB-first
    /// (code 0's K bits at the TOP of word 0). Obviously correct by
    /// construction; the anchor the two transliterations below are checked
    /// against.
    fn pack_stream_u32(codes: &[u16], bits: u32) -> Vec<u32> {
        assert_eq!(codes.len(), 256);
        let nbits = 256 * bits as usize;
        let mut words = vec![0u32; nbits / 32];
        for (u, c) in codes.iter().enumerate() {
            for j in 0..bits as usize {
                // Code bit j (MSB-first: j=0 is the code's top bit) lives at
                // stream bit u*K + j; stream bit k is word bit 31 - (k % 32).
                if (c >> (bits as usize - 1 - j)) & 1 == 1 {
                    let k = u * bits as usize + j;
                    words[k / 32] |= 1 << (31 - (k % 32));
                }
            }
        }
        words
    }

    /// `unpack_trellis_kernel` transliteration over a u32 stream (`pack.cu`):
    /// thread `t` yields windows `2t, 2t+1` (`word01 = w1 << 16 | w0`, so
    /// `unpacked[2t] = w0`, `unpacked[2t+1] = w1`).
    ///
    /// Arg order note: unpack calls `__funnelshift_r(b, a, s1)` with `a` the
    /// FIRST-bit word and `b` the LAST-bit word — the same convention as the
    /// scalar `dq` (`__funnelshift_r(b, a, s0)`), i.e. CUDA's `(lo, hi)` args
    /// map to this module's `funnel(hi, lo, shift)`, so `funnel(a, b, s1)`.
    fn unpack_from_u32(words32: &[u32], bits: u32) -> Vec<u16> {
        let k = bits as usize;
        let n32 = words32.len() as u32;
        let mut out = vec![0u16; 256];
        for t in 0..128usize {
            let b0 = t * 2 * k + k + 256 * k - 16;
            let b1 = b0 + k;
            let b2 = b1 + 16;
            let i0 = b0 / 32;
            let i1 = (b2 - 1) / 32;
            let s1 = ((i1 + 1) * 32 - b2) as u32;
            let a = words32[i0 % n32 as usize];
            let b = words32[i1 % n32 as usize];
            let w1 = funnel(a, b, s1) & 0xffff;
            let w0 = funnel(a, b, s1 + k as u32) & 0xffff;
            out[2 * t] = w0 as u16;
            out[2 * t + 1] = w1 as u16;
        }
        out
    }

    /// Three upstream kernels must agree on the stream layout, or the oracle's
    /// bit order matches neither the quantizer nor the decoder:
    ///
    /// (a) `pack_trellis_kernel`'s u16 output, read as raw u32 pairs (exactly
    ///     what `unpack_trellis_kernel`'s `((uint32_t*) g_packed)[t]` sees,
    ///     SWAP16 included), must equal the reference bitstream — pins the
    ///     SWAP16 endianness dance.
    /// (b) `unpack_trellis_kernel`'s windows must equal the scalar funnel
    ///     reader at stream words `2t, 2t+1` — pins the unpack index
    ///     arithmetic (`i0/i1/s1`) against the Phase-0-verified reader.
    /// (c) Scalar window `s` must carry code `s` in its LOW `K` bits (window
    ///     `[sK+K-16+256K, +16)` overlaps code `[sK, sK+K)` at its tail) — pins
    ///     the code/window alignment the trellis constraint relies on.
    #[test]
    fn pack_unpack_roundtrip_matches_scalar() {
        for bits in 1..=8u32 {
            let mut s = 0x5eed_5eed_5eed_5eed ^ (bits as u64);
            let mask = (1u32 << bits) - 1;
            let codes: Vec<u16> = (0..256).map(|_| (xrs(&mut s) & mask) as u16).collect();
            // (a) transliterated pack (with SWAP16) == reference bitstream.
            let packed = pack_codes(&codes, bits);
            let as_u32: Vec<u32> = packed
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| (p[0] as u32) | ((p[1] as u32) << 16))
                .collect();
            assert_eq!(
                as_u32,
                pack_stream_u32(&codes, bits),
                "bits={bits}: pack transliteration != reference bitstream"
            );
            // (b) unpack transliteration == scalar reader, every window.
            let words32 = pack_stream_u32(&codes, bits);
            let unpacked = unpack_from_u32(&words32, bits);
            for (u, w) in unpacked.iter().enumerate() {
                let want = window16_funnel(&words32, scalar_b0(bits, u as u32));
                assert_eq!(*w, want, "bits={bits} word={u}: unpack != scalar");
            }
            // (c) window s carries code s in its low K bits.
            for (u, w) in unpacked.iter().enumerate() {
                assert_eq!(
                    *w & (mask as u16),
                    codes[u],
                    "bits={bits} word={u}: window does not contain its code"
                );
            }
        }
    }
    /// `decode_matrix` must equal per-tile `decode_tile` scattered by
    /// [`tile_perm`], and stay finite over adversarial bytes at every rate.
    #[test]
    fn decode_matrix_agrees_with_scattered_tiles() {
        for bits in 1..=8u32 {
            let (k, n) = (32, 48);
            let w = tile_words_u16(bits);
            let mut s = 0x7777_7777_7777_7777 ^ (bits as u64);
            let trellis: Vec<u16> = (0..(k / 16) * (n / 16) * w)
                .map(|_| xrs(&mut s) as u16)
                .collect();
            for cb in [Codebook::Cb0, Codebook::Cb1, Codebook::Cb2] {
                let mat = decode_matrix(cb, bits, &trellis, k, n);
                assert_eq!(mat.len(), k * n);
                assert!(mat.iter().all(|v| v.is_finite()), "bits={bits} {cb:?}");
                // Manual scatter of the same tiles.
                let perm = tile_perm(bits);
                let tile32n = w / 2;
                for ki in 0..k / 16 {
                    for nj in 0..n / 16 {
                        let base = (ki * (n / 16) + nj) * w;
                        let t32: Vec<u32> = trellis[base..base + w]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|p| (p[0] as u32) | ((p[1] as u32) << 16))
                            .collect();
                        assert_eq!(t32.len(), tile32n);
                        let vals = decode_tile(cb, bits, &t32);
                        for (st, v) in vals.iter().enumerate() {
                            let p = perm[st] as usize;
                            assert_eq!(
                                mat[(ki * 16 + p / 16) * n + nj * 16 + p % 16],
                                *v,
                                "bits={bits} tile=({ki},{nj}) stream={st}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// `apply_scales` multiplies rows by `suh` and columns by `svh`.
    #[test]
    fn apply_scales_folds_both_sides() {
        let (k, n) = (32, 16);
        let mut mat = vec![1.0f32; k * n];
        let suh: Vec<f32> = (0..k).map(|r| (r + 1) as f32).collect();
        let svh: Vec<f32> = (0..n).map(|c| 0.5 + 0.25 * c as f32).collect();
        apply_scales(&mut mat, &suh, &svh, k, n);
        for r in 0..k {
            for c in 0..n {
                assert_eq!(mat[r * n + c], (r + 1) as f32 * (0.5 + 0.25 * c as f32));
            }
        }
    }

    /// `hadamard128` is the orthonormal FWHT: involution, norm-preserving,
    /// with the two closed forms (`H(e0) = 1/sqrt(128)`, `H(ones)[0] = sqrt(128)`).
    #[test]
    fn hadamard128_is_orthonormal() {
        // e0 -> uniform 1/sqrt(128).
        let mut v = [0.0f32; 128];
        v[0] = 1.0;
        hadamard128(&mut v);
        let want = 1.0 / 128.0f32.sqrt();
        for x in v {
            assert!((x - want).abs() < 1e-6, "H(e0) = {x}, want {want}");
        }
        // ones -> (sqrt(128), 0, ..., 0).
        let mut v = [1.0f32; 128];
        hadamard128(&mut v);
        assert!((v[0] - 128.0f32.sqrt()).abs() < 1e-4, "H(1)[0] = {}", v[0]);
        for x in v.iter().skip(1) {
            assert!(x.abs() < 1e-4, "H(1)[i!=0] = {x}");
        }
        // Involution + norm preservation on PRBS data.
        let mut s = 0x5a5a_5a5a_5a5a_5a5a;
        let orig: Vec<f32> = (0..128)
            .map(|_| (xrs(&mut s) as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();
        let norm0: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut v = orig.clone();
        hadamard128(&mut v);
        let norm1: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm0 - norm1).abs() / norm0 < 1e-5,
            "norm {norm0} -> {norm1}"
        );
        hadamard128(&mut v);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-4, "H(H(x)) != x: {a} vs {b}");
        }
    }
} // mod tests
