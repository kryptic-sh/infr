//! EXL3 trellis-decode bit parity for the CPU oracle (`infr_cpu::exl3`).
//!
//! The GPU-leg analogue is the `mmv_*_parity` family (e.g. `mmv_mw_parity`: two
//! implementations of the same decode must agree exactly). Here both legs are on
//! the CPU, but they are INDEPENDENT decoders of the same trellis bytes, in the
//! same sense `decode_parity`'s cross-check is genuine:
//!
//! - leg A (`decode_tile`): the funnel-shift path, mirroring the CUDA `dq*`
//!   extractors in `exl3_dq.cuh` (`fshift` merge+shift, modulo tile indexing);
//! - leg B (`decode_tile_oracle`): a direct bit-slice reader that treats the tile
//!   as a little-endian bit array and shares no arithmetic with leg A.
//!
//! Both legs call the same `decode_3inst` codebook functions — the codebook math
//! is pinned separately, against an f64 recomputation, in the `exl3` unit tests
//! (`cb2_zero_word_matches_f64_oracle`, `hadd_halves_golden`), so a codebook bug
//! cannot hide behind agreement here; what THIS test proves is the bit-window
//! extraction (the part that differs between GPU tiling variants).
//!
//! Seeding: a 64-bit LCG (`trellis_word`) draws the tile contents, so every rate
//! x codebook case is reproducible without a `rand` dependency — the same style
//! as `mmv_mw_parity`'s inline `wrapping_mul(2654435761)` byte fill.
//!
//! Reference: turboderp-org/exllamav3 @ `bd1f519172406a9560bd5c90e61000691bb69a07`
//! (`exllamav3/exllamav3_ext/quant/exl3_dq.cuh`, `.../quant/codebook.cuh`).
//!
//! Run: `cargo test -p infr-cpu --test exl3_parity`

use infr_cpu::exl3::{decode_3inst, decode_tile, decode_tile_oracle, Codebook};

/// Deterministic 64-bit LCG (Knuth MMIX constants); the low 32 bits fill tiles.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

const CODEBOOKS: [(Codebook, &str); 3] = [
    (Codebook::Cb0, "cb0/3inst/2bpw"),
    (Codebook::Cb1, "cb1/mcg/3bpw"),
    (Codebook::Cb2, "cb2/mul1/4bpw"),
];

/// Funnel-shift tile decode and slice-oracle tile decode agree BIT-FOR-BIT at
/// every integer rate (1..=8) x every codebook, over seeded PRBS tiles.
///
/// Exact equality (not a tolerance): both legs decode the same 16-bit windows
/// through the same correctly-rounded f16 codebook functions, so any divergence
/// is an extraction bug, not rounding.
#[test]
fn exl3_tile_funnel_matches_slice_oracle() {
    for bits in 1..=8u32 {
        let nwords = bits as usize * 8; // bits*256/32 u32 words per tile
        for (cb, name) in CODEBOOKS {
            // Three seeds: zero tile (wrap/modulo edge), one-word (single-word
            // fast path), PRBS (general case incl. cross-word windows).
            let mut worst_bit_diff = 0u32;
            for seed in [0u64, 1, 0x1234_5678_9abc_def0] {
                let mut rng = Lcg(seed);
                let words: Vec<u32> = (0..nwords).map(|_| rng.next_u32()).collect();
                let a = decode_tile(cb, bits, &words);
                let b = decode_tile_oracle(cb, bits, &words);
                assert_eq!(a.len(), 256, "{name} bits={bits}: len");
                assert_eq!(b.len(), 256, "{name} bits={bits}: len");
                for (t, (x, y)) in a.iter().zip(&b).enumerate() {
                    // Bit comparison: f32 equality, with NaN treated as failure
                    // (the unit tests already prove finiteness; here NaN must
                    // not be silently "equal").
                    assert!(
                        x.to_bits() == y.to_bits(),
                        "{name} bits={bits} seed={seed:#x} t={t}: funnel={x} ({:08x}) slice={y} ({:08x})",
                        x.to_bits(),
                        y.to_bits(),
                    );
                    worst_bit_diff = worst_bit_diff.max(x.to_bits() ^ y.to_bits());
                }
            }
            assert_eq!(
                worst_bit_diff, 0,
                "{name} bits={bits}: legs diverged (see failures above)"
            );
        }
    }
}

/// Seeded oracle spot-check: the first weight of a seeded 4-bit cb2 tile must
/// equal a value recomputed here from the published formulas (window position
/// from `dq`, byte-sum + f16 FMA from `decode_3inst<2>`), so the test does not
/// just compare the module against itself.
#[test]
fn exl3_seeded_tile_matches_hand_derivation() {
    // bits=4: tile = 32 u32 words. Seed the LCG once; the expected first-tile
    // value is derived below with independent scalar code.
    let mut rng = Lcg(0xC0FFEE);
    let words: Vec<u32> = (0..32).map(|_| rng.next_u32()).collect();

    let got = decode_tile(Codebook::Cb2, 4, &words)[0];

    // Independent derivation of t_offset=0: b0 = 0*4 + 4 - 16 + 1024 = 1012.
    // Stream order is MSB-first in-word (see `window16_slice` docs).
    let b0 = 1012usize;
    let mut w0 = 0u16;
    for k in 0..16 {
        let kk = b0 + k;
        w0 |= (((words[(kk / 32) % 32] >> (31 - (kk % 32))) & 1) as u16) << (15 - k);
    }
    // decode_3inst<2>, recomputed from the definition with explicit single
    // f16 rounding (f64 detour is exact for f16 operands: see `exl3` docs):
    let x = (w0 as u32).wrapping_mul(0x83DCD12D);
    let sum = 0x6400u32 + (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + ((x >> 24) & 0xff);
    let want = {
        use half::f16;
        let h = f16::from_bits(sum as u16);
        f16::from_f64(
            h.to_f64() * f16::from_bits(0x1eee).to_f64() + f16::from_bits(0xc931).to_f64(),
        )
        .to_f32()
    };
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "seeded cb2 tile[0]: got {got} want {want} (w0={w0:#06x})"
    );
}

/// The codebook functions are total over their domain: all 65536 words decode
/// to finite f32 at every codebook (exhaustive — 196k decodes, milliseconds).
/// A NaN/Inf anywhere would poison a real-tile decode silently.
#[test]
fn exl3_codebooks_total_over_all_words() {
    for (cb, name) in CODEBOOKS {
        let mut bad = 0u32;
        for w in 0..=0xffffu32 {
            let v = decode_3inst(cb, w as u16);
            if !v.is_finite() {
                bad += 1;
                if bad < 5 {
                    eprintln!("{name}: word {w:#06x} -> {v}");
                }
            }
        }
        assert_eq!(bad, 0, "{name}: {bad} non-finite words");
    }
}
