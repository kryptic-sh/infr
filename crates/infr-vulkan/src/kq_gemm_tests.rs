//! Kernel-level parity for the K-quant prefill GEMMs that read int8 activations: the int8
//! cooperative-matrix one (`native_gemm_i8cm_kq.comp`,
//! [`crate::recorder::Recorder::matmul_i8cm_kq`]) and the packed-int8-dot one
//! (`native_gemm_dp4a_kq.comp`, [`crate::recorder::Recorder::matmul_dp4a_kq`]).
//!
//! The reference is the kernels' own decomposition evaluated in f64 on the host, from the GPU's
//! downloaded int8 activation codes and their f16 scales: per 32-element block,
//! `dA · Σ qa·(w − min) + sA · min`, with `w` from the production host dequant and `min` the same
//! dequant of the block with its quant bits zeroed. The only freedom left is f32 association, so
//! the tolerance is a small multiple of f32 epsilon over the absolute term sum; a wrong nibble,
//! scale, sign or row lands at the size of the terms themselves. What int8 activations cost
//! against f32 is bounded separately, in `seam_op_parity`'s `kquant_linear_*` tests.
//!
//! On activations made only of small blocks, whose f16 scales are subnormal, the same
//! decomposition also holds the dense `mmq` GEMM and the `mmv_mw` decode GEMV, and `quant_q8`'s
//! stored scales are held to f16's rounding of `amax / 127`: what a driver that flushes f16
//! subnormals breaks.
//!
//! The same file holds the routed experts' outlier quantizer (`quant_q8.comp`'s `-DHI` build) to
//! its rule, and the batched expert GEMMs that read it (`-DACT_HI`) to their decomposition plus
//! each block's carried remainder.
// Wrapped in its own `#[cfg(test)]` module, like every in-crate test, so the bare-print guard
// (`infr-core/tests/no_bare_print.rs`) reads its skip notices as test output.
#[cfg(test)]
mod tests {
    use crate::gemm::{I8cmTile, DP4A_KQ_TILE};
    use crate::recorder::Recorder;
    use crate::VulkanBackend;
    use infr_core::backend::{Backend, Buffer, BufferUsage};
    use infr_core::DType;

    /// Tokens: two large tiles, the second partial, and a partial medium tile.
    const M: usize = 200;
    /// Two superblocks, so a weight row spans more than one superblock.
    const K: usize = 512;
    /// One large tile, two medium ones.
    const N: usize = 128;
    /// One superblock of other weights ahead of the tested rows: exercises `w_base`, and puts the
    /// Q6_K superblocks (210 bytes) on 2-byte boundaries.
    const W_BASE: usize = 256;
    /// Output rows: two extra rows of sentinel past M, where a store past the last real row shows.
    const MPAD: usize = M + 2;

    struct Rng(u64);
    impl Rng {
        fn byte(&mut self) -> u8 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u8
        }
    }

    /// A valid bank of `n_elems` weights: random quant and scale bits with small f16 super-scales.
    fn synth_bank(dt: DType, n_elems: usize, seed: u64) -> Vec<u8> {
        let (epb, bpb) = infr_gguf::block_layout(dt);
        let mut rng = Rng(seed);
        let mut out = Vec::with_capacity(n_elems / epb * bpb);
        for _ in 0..n_elems / epb {
            let mut b: Vec<u8> = (0..bpb).map(|_| rng.byte()).collect();
            let d = half::f16::from_f32(0.004 + (rng.byte() as f32) * 1e-5).to_le_bytes();
            let dmin = half::f16::from_f32(0.002 + (rng.byte() as f32) * 1e-5).to_le_bytes();
            match dt {
                DType::Q4K | DType::Q5K => {
                    b[0..2].copy_from_slice(&d);
                    b[2..4].copy_from_slice(&dmin);
                }
                DType::Q6K => b[208..210].copy_from_slice(&d),
                other => panic!("no synth for {other:?}"),
            }
            out.extend_from_slice(&b);
        }
        out
    }

    /// The per-block additive term of `bank` (`−dmin·mn` per 32 elements), or zeros for Q6_K, whose
    /// values are symmetric: the dequant of the same bank with every quant bit cleared.
    fn block_mins(dt: DType, bank: &[u8]) -> Vec<f32> {
        let (_, bpb) = infr_gguf::block_layout(dt);
        let quant_bytes = match dt {
            DType::Q4K => 16..144,
            DType::Q5K => 16..176,
            DType::Q6K => return vec![0.0; bank.len() / bpb * 8],
            other => panic!("no mins for {other:?}"),
        };
        let mut zeroed = bank.to_vec();
        for block in zeroed.chunks_mut(bpb) {
            block[quant_bytes.clone()].fill(0);
        }
        let w = infr_gguf::dequant::dequant_block(dt, &zeroed).unwrap();
        w.chunks(32).map(|c| c[0]).collect()
    }

    fn f16_at(bytes: &[u8], i: usize) -> f64 {
        half::f16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]).to_f64()
    }

    fn download_bytes(be: &VulkanBackend, b: &dyn Buffer) -> Vec<u8> {
        let mut v = vec![0u8; b.len_bytes()];
        be.download(b, &mut v).unwrap();
        v
    }

    /// The activations and their int8 quantization, shared by every GEMM run of a test.
    struct Fixture {
        be: VulkanBackend,
        xbuf: Box<dyn Buffer>,
        qa: Box<dyn Buffer>,
        dact: Box<dyn Buffer>,
        sact: Box<dyn Buffer>,
    }

    impl Fixture {
        fn new(be: VulkanBackend) -> Self {
            let x: Vec<f32> = (0..M * K)
                .map(|i| (i as f32 * 0.37).sin() * (1.0 + (i % 97) as f32 * 0.05))
                .collect();
            Self::with_x(be, &x)
        }

        /// The fixture on activations `x` (`M x K`).
        fn with_x(be: VulkanBackend, x: &[f32]) -> Self {
            assert_eq!(x.len(), M * K);
            let xbuf = be.alloc(M * K * 4, BufferUsage::Activations).unwrap();
            be.upload(xbuf.as_ref(), bytemuck::cast_slice(x)).unwrap();
            let nblk = K / 32;
            let qa = be.alloc(M * K, BufferUsage::Activations).unwrap();
            let dact = be.alloc(M * nblk * 2, BufferUsage::Activations).unwrap();
            let sact = be.alloc(M * nblk * 2, BufferUsage::Activations).unwrap();
            Fixture {
                be,
                xbuf,
                qa,
                dact,
                sact,
            }
        }

        /// A weight bank for `dt`: `W_BASE` elements of other weights, then the `N x K` tested
        /// rows. Returns the uploaded bank, the host dequant of the tested rows and their block
        /// mins.
        fn weights(&self, dt: DType) -> (Box<dyn Buffer>, Vec<f32>, Vec<f32>) {
            let (epb, bpb) = infr_gguf::block_layout(dt);
            let mut bank = synth_bank(dt, W_BASE, 0xbad0 ^ dt as u64);
            let rows = synth_bank(dt, N * K, 0x5eed ^ dt as u64);
            bank.extend_from_slice(&rows);
            let w_host = infr_gguf::dequant::dequant_block(dt, &rows).unwrap();
            let mins = block_mins(dt, &rows);
            let wbuf = self.be.alloc(bank.len(), BufferUsage::Weights).unwrap();
            self.be.upload(wbuf.as_ref(), &bank).unwrap();
            assert_eq!(bank.len(), (W_BASE + N * K) / epb * bpb);
            (wbuf, w_host, mins)
        }

        /// Quantize the activations, record `gemm` into the same submission writing a
        /// sentinel-filled output, and return that `MPAD x N` output.
        fn run(&self, gemm: impl FnOnce(&Recorder<'_>, &dyn Buffer)) -> Vec<f32> {
            let be = &self.be;
            let out = be.alloc(MPAD * N * 4, BufferUsage::Activations).unwrap();
            let sentinel = vec![f32::NAN; MPAD * N];
            be.upload(out.as_ref(), bytemuck::cast_slice(&sentinel))
                .unwrap();
            let rec = be.recorder().unwrap();
            rec.quant_q8(
                self.xbuf.as_ref(),
                self.qa.as_ref(),
                self.dact.as_ref(),
                self.sact.as_ref(),
                M,
                K,
            );
            gemm(&rec, out.as_ref());
            rec.finish().unwrap();
            let mut got = vec![0f32; MPAD * N];
            be.download(out.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            got
        }

        /// Hold `got` (an `MPAD x N` output) to the decomposition from the GPU's own int8 codes,
        /// and the rows past `M` to their sentinel.
        fn assert_matches_decomposition(
            &self,
            dt: DType,
            what: &str,
            got: &[f32],
            w_host: &[f32],
            mins: &[f32],
        ) {
            let host = self.host();
            let mut worst = 0f64;
            for r in 0..M {
                for o in 0..N {
                    let (want, mag) = decomposition(&host, r, o, w_host, mins);
                    let g = got[r * N + o] as f64;
                    let err = (g - want).abs();
                    worst = worst.max(err / mag);
                    assert!(
                        err <= 1e-5 * mag,
                        "{dt:?} {what}: row {r} col {o}: got {g} want {want} (term sum {mag})"
                    );
                }
            }
            eprintln!("{dt:?} {what}: worst error / term sum = {worst:e}");
            assert!(
                got[M * N..].iter().all(|v| v.is_nan()),
                "{dt:?} {what}: wrote rows past m"
            );
        }
    }

    impl Fixture {
        /// The GPU's int8 codes, f16 scales and f16 sums, downloaded.
        fn host(&self) -> [Vec<u8>; 3] {
            [&self.qa, &self.dact, &self.sact].map(|b| download_bytes(&self.be, b.as_ref()))
        }
    }

    /// Output `(r, o)` from the GPU's own codes, scales and sums (`host`, as [`Fixture::host`]
    /// returns them), evaluated in f64, and the absolute term sum it is held to.
    fn decomposition(
        host: &[Vec<u8>; 3],
        r: usize,
        o: usize,
        w_host: &[f32],
        mins: &[f32],
    ) -> (f64, f64) {
        let [qa_h, da_h, sa_h] = host;
        let nblk = K / 32;
        let (mut want, mut mag) = (0f64, 0f64);
        for b in 0..nblk {
            let da = f16_at(da_h, r * nblk + b);
            let sa = f16_at(sa_h, r * nblk + b);
            let min = mins[o * nblk + b] as f64;
            let mut dot = 0f64;
            for i in b * 32..(b + 1) * 32 {
                let q = qa_h[r * K + i] as i8 as f64;
                let t = q * (w_host[o * K + i] as f64 - min);
                dot += t;
                mag += (t * da).abs();
            }
            want += da * dot + sa * min;
            mag += (sa * min).abs();
        }
        (want, mag)
    }

    /// Run every K-quant at both tiles; each output checked against the decomposition, both tiles
    /// bitwise equal (the k order is the same, so the tile choice can never move a result), rows past
    /// `M` untouched.
    #[test]
    #[ignore = "requires a Vulkan GPU with int8 cooperative matrix"]
    fn i8cm_kq_matches_its_decomposition() {
        let Some(be) = infr_testkit::gpu_or_skip_noted(VulkanBackend::new()) else {
            return;
        };
        if !be.i8cm_kq_ready() {
            eprintln!(
                "i8cm_kq_matches_its_decomposition: int8 coopmat GEMM not armed here — skipped"
            );
            return;
        }
        let fx = Fixture::new(be);
        for dt in [DType::Q4K, DType::Q5K, DType::Q6K] {
            let (wbuf, w_host, mins) = fx.weights(dt);
            let w_addr = wbuf.device_addr().expect("weights carry a device address");
            let outs: Vec<Vec<f32>> = [I8cmTile::L, I8cmTile::M]
                .into_iter()
                .map(|tile| {
                    fx.run(|rec, out| {
                        rec.matmul_i8cm_kq(
                            dt,
                            tile,
                            fx.qa.as_ref(),
                            fx.dact.as_ref(),
                            fx.sact.as_ref(),
                            w_addr,
                            W_BASE,
                            out,
                            M,
                            K,
                            N,
                        )
                    })
                })
                .collect();
            for (out, tile) in outs.iter().zip(["large tile", "medium tile"]) {
                fx.assert_matches_decomposition(dt, tile, out, &w_host, &mins);
            }
            let ndiff = outs[0][..M * N]
                .iter()
                .zip(&outs[1][..M * N])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                ndiff, 0,
                "{dt:?}: the two tiles disagree in {ndiff} outputs"
            );
        }
    }

    /// The packed-int8-dot GEMM for every K-quant, against the same decomposition. It needs only
    /// int8 dot, so it runs on any device that has that; the shape takes a partial last token tile
    /// and more than one weight-row tile.
    #[test]
    #[ignore = "requires a Vulkan GPU with int8 dot"]
    fn dp4a_kq_matches_its_decomposition() {
        let Some(be) = infr_testkit::gpu_or_skip_noted(VulkanBackend::new()) else {
            return;
        };
        if !be.caps().i8_dot {
            eprintln!("dp4a_kq_matches_its_decomposition: no int8 dot here — skipped");
            return;
        }
        assert!(M > DP4A_KQ_TILE.tokens && !M.is_multiple_of(DP4A_KQ_TILE.tokens));
        assert!(N > DP4A_KQ_TILE.rows && N.is_multiple_of(DP4A_KQ_TILE.rows));
        let fx = Fixture::new(be);
        for dt in [DType::Q4K, DType::Q5K, DType::Q6K] {
            let (wbuf, w_host, mins) = fx.weights(dt);
            let w_addr = wbuf.device_addr().expect("weights carry a device address");
            let got = fx.run(|rec, out| {
                rec.matmul_dp4a_kq(
                    dt,
                    fx.qa.as_ref(),
                    fx.dact.as_ref(),
                    fx.sact.as_ref(),
                    w_addr,
                    W_BASE,
                    out,
                    M,
                    K,
                    N,
                )
            });
            fx.assert_matches_decomposition(dt, "dp4a", &got, &w_host, &mins);
        }
    }

    /// Block sizes of [`small_x`]: every one under `127 · 2^-14 ≈ 7.75e-3`, so every scale
    /// `quant_q8` writes for them is an f16 subnormal, down to one that rounds to f16's least
    /// subnormal value.
    const SMALL_AMAX: [f32; 8] = [7e-3, 3e-3, 1e-3, 3e-4, 1e-4, 5e-5, 2e-5, 1e-5];

    /// `M x K` activations made only of small blocks, as gemma-3 has many of: block `b` of row
    /// `r` reaches `SMALL_AMAX[(r + b) % 8]` at its lane 0 and stays within it elsewhere.
    fn small_x() -> Vec<f32> {
        (0..M * K)
            .map(|i| {
                let (r, k) = (i / K, i % K);
                let amax = SMALL_AMAX[(r + k / 32) % SMALL_AMAX.len()];
                match k % 32 {
                    0 => amax,
                    lane => amax * (lane as f32 * 0.61 + r as f32 * 1.7).cos(),
                }
            })
            .collect()
    }

    /// The int8 kernels keep a block whose scale is an f16 subnormal: `quant_q8` writes each such
    /// scale as f16 rounds `amax / 127` (to the nearest, within one unit of the last place for the
    /// device's own division), never zero, and the dense `mmq` GEMM (the non-coopmat tier's
    /// prefill GEMM), the dp4a K-quant GEMM and the decode GEMV (`mmv_mw`) each match the f64
    /// decomposition from those very codes and scales. A driver that flushes f16 subnormals to
    /// zero, which SPIR-V allows where no float-controls mode is declared, fails the first check
    /// where it stores the scale and the decompositions where it reads one: such a block then drops
    /// out of the dot, and every row here is made of such blocks. On a device that keeps them both
    /// pass; this is a check of the driver's f16 handling under the kernels as built.
    #[test]
    #[ignore = "requires a Vulkan GPU with int8 dot"]
    fn int8_kernels_keep_f16_subnormal_scales() {
        let Some(be) = infr_testkit::gpu_or_skip_noted(VulkanBackend::new()) else {
            return;
        };
        if !be.caps().i8_dot {
            eprintln!("int8_kernels_keep_f16_subnormal_scales: no int8 dot here — skipped");
            return;
        }
        let x = small_x();
        let fx = Fixture::with_x(be, &x);
        let be = &fx.be;
        let nblk = K / 32;
        for dt in [DType::Q4K, DType::Q6K] {
            let (wbuf, w_host, mins) = fx.weights(dt);
            let w_addr = wbuf.device_addr().expect("weights carry a device address");
            let got = fx.run(|rec, out| {
                rec.matmul_dp4a_kq(
                    dt,
                    fx.qa.as_ref(),
                    fx.dact.as_ref(),
                    fx.sact.as_ref(),
                    w_addr,
                    W_BASE,
                    out,
                    M,
                    K,
                    N,
                )
            });
            // The scales as stored: f16's rounding of amax / 127, subnormal and not zero.
            let host = fx.host();
            for r in 0..M {
                for b in 0..nblk {
                    let blk = &x[r * K + b * 32..r * K + (b + 1) * 32];
                    let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let want = half::f16::from_f32(amax / 127.0);
                    assert!(
                        !want.is_normal() && want.to_bits() & 0x7fff != 0,
                        "row {r} block {b}: not a subnormal case"
                    );
                    let i = r * nblk + b;
                    let got = u16::from_le_bytes([host[1][2 * i], host[1][2 * i + 1]]);
                    assert!(
                        got & 0x7fff != 0 && got.abs_diff(want.to_bits()) <= 1,
                        "row {r} block {b}: scale stored as {:e} (bits {got:#06x}), \
                         f16 rounds amax / 127 to {:e} (bits {:#06x})",
                        half::f16::from_bits(got).to_f64(),
                        want.to_f64(),
                        want.to_bits()
                    );
                }
            }
            fx.assert_matches_decomposition(dt, "dp4a, small blocks", &got, &w_host, &mins);

            // The dense mmq GEMM writes whole 64-row tiles: an output of its own, rows past M
            // unchecked.
            let mt = M.div_ceil(64) * 64;
            let out = be.alloc(mt * N * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.matmul_mmq_at(
                dt,
                fx.qa.as_ref(),
                fx.dact.as_ref(),
                fx.sact.as_ref(),
                w_addr,
                W_BASE,
                out.as_ref(),
                M,
                K,
                N,
            );
            rec.finish().unwrap();
            let mut mmq = vec![0f32; mt * N];
            be.download(out.as_ref(), bytemuck::cast_slice_mut(&mut mmq))
                .unwrap();
            // The decode GEMV on row 0 (its activation is the first row of the fixture's).
            let y = be.alloc(N * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.linear_mmv_mw(
                dt,
                4,
                wbuf.as_ref(),
                W_BASE,
                fx.qa.as_ref(),
                fx.dact.as_ref(),
                fx.sact.as_ref(),
                None,
                y.as_ref(),
                K,
                N,
            );
            rec.finish().unwrap();
            let mut gemv = vec![0f32; N];
            be.download(y.as_ref(), bytemuck::cast_slice_mut(&mut gemv))
                .unwrap();
            for (what, rows, out) in [("mmq", M, &mmq), ("mmv_mw", 1, &gemv)] {
                let mut worst = 0f64;
                for r in 0..rows {
                    for o in 0..N {
                        let (want, mag) = decomposition(&host, r, o, &w_host, &mins);
                        let g = out[r * N + o] as f64;
                        let err = (g - want).abs();
                        worst = worst.max(err / mag);
                        assert!(
                            err <= 1e-5 * mag,
                            "{dt:?} {what}, small blocks: row {r} col {o}: got {g} want {want} \
                             (term sum {mag})"
                        );
                    }
                }
                eprintln!("{dt:?} {what}, small blocks: worst error / term sum = {worst:e}");
            }
        }
    }

    // ── The routed experts' outlier quantizer (`quant_q8.comp`'s `-DHI`) and the expert GEMMs
    // that read it (`-DACT_HI`), against their rule and an f64 reference. ──

    /// The outlier tests' depth: a multiple of the K-quant superblock, more than one of them, so
    /// a weight row spans superblocks.
    const K_HI: usize = 512;

    /// `rows` activation rows of `K_HI`, ordinary values (`|x| <= 0.15`) with the cases the
    /// outlier quantizer exists for or must leave alone, by row:
    /// - `r % 4 == 0`: one massive activation (±150) in block `r % 16`, at lane `(7r + 1) % 32`
    ///   (`≡ 1 mod 4`, the second byte of a packed word);
    /// - `r % 4 == 1`: nothing but ordinary values;
    /// - `r % 4 == 2`: massive activations in two blocks (80 in block 3, −60 in block 9);
    /// - `r % 4 == 3`: block 5's largest exactly `QUANT_Q8_HI_RATIO` times the next (does not
    ///   split), block 6's just over it (splits), block 7 all zero, block 8 a single non-zero
    ///   value, block 10 two equal largest values (does not split).
    fn hi_rows(rows: usize) -> Vec<f32> {
        let mut x: Vec<f32> = (0..rows * K_HI)
            .map(|i| ((i % K_HI) as f32 * 0.37 + (i / K_HI) as f32 * 1.3).sin() * 0.15)
            .collect();
        let ratio = crate::recorder::QUANT_Q8_HI_RATIO;
        for (r, row) in x.chunks_mut(K_HI).enumerate() {
            match r % 4 {
                0 => {
                    let sign = if r % 8 == 0 { 1.0 } else { -1.0 };
                    row[(r % 16) * 32 + (7 * r + 1) % 32] = 150.0 * sign;
                }
                2 => {
                    row[3 * 32 + 11] = 80.0;
                    row[9 * 32 + 30] = -60.0;
                }
                3 => {
                    // The largest magnitude of block `b` without lane `lane`.
                    let next = |row: &[f32], b: usize, lane: usize| {
                        (0..32)
                            .filter(|&i| i != lane)
                            .fold(0f32, |m, i| m.max(row[b * 32 + i].abs()))
                    };
                    row[5 * 32 + 9] = ratio * next(row, 5, 9);
                    row[6 * 32 + 2] = ratio * next(row, 6, 2) * 1.001;
                    row[7 * 32..8 * 32].fill(0.0);
                    row[8 * 32..9 * 32].fill(0.0);
                    row[8 * 32 + 4] = 0.7;
                    row[10 * 32 + 1] = 5.0;
                    row[10 * 32 + 20] = -5.0;
                }
                _ => {}
            }
        }
        x
    }

    /// The rule `quant_q8.comp`'s `-DHI` build applies to one block: the lowest lane holding the
    /// largest magnitude, and whether that magnitude stands more than `QUANT_Q8_HI_RATIO` times
    /// above every other one.
    fn hi_split(blk: &[f32]) -> (usize, bool) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let top = blk.iter().position(|v| v.abs() == amax).unwrap();
        let next = (0..blk.len())
            .filter(|&i| i != top)
            .fold(0f32, |m, i| m.max(blk[i].abs()));
        (top, amax > crate::recorder::QUANT_Q8_HI_RATIO * next)
    }

    fn vec2_at(bytes: &[u8], i: usize) -> (f32, f32) {
        let f = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        (f(8 * i), f(8 * i + 4))
    }

    fn upload_f32(be: &VulkanBackend, v: &[f32]) -> Box<dyn Buffer> {
        let b = be.alloc(v.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
        b
    }

    fn upload_u32(be: &VulkanBackend, v: &[u32]) -> Box<dyn Buffer> {
        let b = be.alloc(v.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
        b
    }

    /// A remainder and lane no real block carries ([`Quantized::new`]).
    const HRES_STALE: [f32; 2] = [1000.0, 5.0];

    /// One quantization's outputs: codes, f16 scales, f16 sums, and (the `-DHI` builds) the
    /// carried remainders.
    struct Quantized {
        qa: Box<dyn Buffer>,
        dact: Box<dyn Buffer>,
        sact: Box<dyn Buffer>,
        hres: Box<dyn Buffer>,
    }

    impl Quantized {
        /// `hres` starts as [`HRES_STALE`] in every block, standing for what a pooled buffer
        /// holds from an earlier layer: the quantizer writes only the blocks that split, so a
        /// consumer that read an unmarked block would pick it up.
        fn new(be: &VulkanBackend, rows: usize, k: usize) -> Self {
            let nblk = k / 32;
            let a = |n: usize| be.alloc(n, BufferUsage::Activations).unwrap();
            Quantized {
                qa: a(rows * k),
                dact: a(rows * nblk * 2),
                sact: a(rows * nblk * 2),
                hres: upload_f32(be, &HRES_STALE.repeat(rows * nblk)),
            }
        }

        fn host(&self, be: &VulkanBackend) -> [Vec<u8>; 4] {
            [&self.qa, &self.dact, &self.sact, &self.hres].map(|b| download_bytes(be, b.as_ref()))
        }
    }

    /// The outlier quantizer against its rule, block by block: a block that does not split gets
    /// the plain build's bits, its scale's sign bit clear; one that splits has the sign bit of its
    /// scale set (the mark the GEMMs read `hres` on), takes the scale of the rest (each other
    /// element within half a step of it), keeps a ±127 code at its outlier, and the code times the
    /// f16 scale plus the remainder is the outlier again. The gather build gives the same bits as
    /// the plain one for the row each slot reads, and the same remainder for each split block.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn quant_q8_hi_follows_its_rule() {
        let Some(be) = infr_testkit::gpu_or_skip_noted(VulkanBackend::new()) else {
            return;
        };
        let rows = 12;
        let nblk = K_HI / 32;
        let x = hi_rows(rows);
        let xbuf = upload_f32(&be, &x);
        // Slot j reads row perm[j]: reversed, so no slot reads its own row.
        let perm: Vec<u32> = (0..rows as u32).rev().collect();
        let pbuf = upload_u32(&be, &perm);
        let (plain, hi, gathered) = (
            Quantized::new(&be, rows, K_HI),
            Quantized::new(&be, rows, K_HI),
            Quantized::new(&be, rows, K_HI),
        );
        let rec = be.recorder().unwrap();
        let x_in = xbuf.as_ref();
        let (p, h, g) = (&plain, &hi, &gathered);
        rec.quant_q8(
            x_in,
            p.qa.as_ref(),
            p.dact.as_ref(),
            p.sact.as_ref(),
            rows,
            K_HI,
        );
        let hres = Some(h.hres.as_ref());
        rec.quant_q8_with(
            x_in,
            h.qa.as_ref(),
            h.dact.as_ref(),
            h.sact.as_ref(),
            hres,
            rows,
            K_HI,
        );
        rec.quant_q8_gather_with(
            x_in,
            pbuf.as_ref(),
            g.qa.as_ref(),
            g.dact.as_ref(),
            g.sact.as_ref(),
            Some(g.hres.as_ref()),
            rows,
            K_HI,
        );
        rec.finish().unwrap();
        assert!(be.built_kernel_names().contains(&"quant_q8_hi"));
        let [qa0, da0, sa0, _] = plain.host(&be);
        let [qa1, da1, sa1, h1] = hi.host(&be);
        let [qa2, da2, sa2, h2] = gathered.host(&be);
        let (mut split_blocks, mut whole_blocks) = (0, 0);
        for r in 0..rows {
            for b in 0..nblk {
                let i = r * nblk + b;
                let blk = &x[r * K_HI + b * 32..r * K_HI + (b + 1) * 32];
                let codes = |qa: &[u8]| qa[r * K_HI + b * 32..r * K_HI + (b + 1) * 32].to_vec();
                let (top, split) = hi_split(blk);
                let marked = da1[2 * i + 1] & 0x80 != 0;
                assert_eq!(marked, split, "row {r} block {b}: the split mark");
                if !split {
                    whole_blocks += 1;
                    assert_eq!(codes(&qa1), codes(&qa0), "row {r} block {b}: codes");
                    assert_eq!(
                        da1[2 * i..2 * i + 2],
                        da0[2 * i..2 * i + 2],
                        "row {r} block {b}"
                    );
                    assert_eq!(
                        sa1[2 * i..2 * i + 2],
                        sa0[2 * i..2 * i + 2],
                        "row {r} block {b}"
                    );
                    continue;
                }
                split_blocks += 1;
                let (rem, lane) = vec2_at(&h1, i);
                let next = (0..32)
                    .filter(|&l| l != top)
                    .fold(0f32, |m, l| m.max(blk[l].abs()));
                let d = f16_at(&da1, i).abs();
                let want_d = next as f64 / 127.0;
                assert!(
                    (d - want_d).abs() <= want_d * 2f64.powi(-10),
                    "row {r} block {b}: scale {d}, the rest's {want_d}"
                );
                assert_eq!(lane, top as f32, "row {r} block {b}: lane");
                let c = codes(&qa1);
                for (l, (&q, &v)) in c.iter().zip(blk).enumerate() {
                    let q = q as i8 as f64;
                    if l == top {
                        let want_q = if next > 0.0 {
                            127.0 * v.signum() as f64
                        } else {
                            0.0
                        };
                        assert_eq!(q, want_q, "row {r} block {b}: the outlier's code");
                        let back = q * d + rem as f64;
                        assert!(
                            (back - v as f64).abs() <= 1e-6 * (v as f64).abs(),
                            "row {r} block {b}: outlier {v} comes back as {back}"
                        );
                    } else {
                        // A split block's scale is f16-exact, so the codes round against the
                        // very scale read back: half a step, and a hair for `v · (1/d)`'s f32
                        // rounding at a tie.
                        assert!(
                            (q * d - v as f64).abs() <= d * (0.5 + 1e-6),
                            "row {r} block {b} lane {l}: {v} quantized to {q} x {d}"
                        );
                    }
                }
            }
        }
        // Every kind of block ran: the massive-activation rows and row 3's just-over block split;
        // the ordinary blocks, the exact-ratio, all-zero and tied blocks did not.
        assert_eq!(split_blocks, rows / 4 * (1 + 2 + 2));
        assert!(whole_blocks > 0);
        for (j, &src) in perm.iter().enumerate() {
            let s = src as usize;
            let row = |v: &[u8], w: usize| v[j * w..(j + 1) * w].to_vec();
            let src_row = |v: &[u8], w: usize| v[s * w..(s + 1) * w].to_vec();
            for (got, want, w) in [
                (&qa2, &qa1, K_HI),
                (&da2, &da1, nblk * 2),
                (&sa2, &sa1, nblk * 2),
            ] {
                assert_eq!(row(got, w), src_row(want, w), "gather slot {j} (row {s})");
            }
            // `hres` holds a value only where a block split (the mark checked above).
            for b in (0..nblk).filter(|&b| da1[2 * (s * nblk + b) + 1] & 0x80 != 0) {
                assert_eq!(
                    vec2_at(&h2, j * nblk + b),
                    vec2_at(&h1, s * nblk + b),
                    "gather slot {j} (row {s}) block {b}: remainder"
                );
            }
        }
        eprintln!("quant_q8_hi: {split_blocks} blocks split, {whole_blocks} whole");
    }

    /// The outlier-reading expert GEMMs (Q4_K, Q6_K, every resident tile) against the f64 sum of
    /// their decomposition from the GPU's own codes, scales, sums and remainders, plus the
    /// remainder times the weight at its lane. Then what it is for: on a row with a massive
    /// activation its output is held to the exact f64 dot of the f32 activation at least ten
    /// times closer than the plain GEMM's, and on a row without one it matches the plain GEMM's
    /// to within 1e-5 of the absolute term sum (the two builds may contract the epilogue
    /// differently). The carried lanes cover every byte position of a packed word.
    #[test]
    #[ignore = "requires a Vulkan GPU with int8 dot"]
    fn expert_gemm_hi_matches_its_decomposition() {
        let Some(be) = infr_testkit::gpu_or_skip_noted(VulkanBackend::new()) else {
            return;
        };
        if !be.caps().i8_dot {
            eprintln!("expert_gemm_hi_matches_its_decomposition: no int8 dot here — skipped");
            return;
        }
        // (rows per expert, n, the tile `matmul_mmq_experts` picks for it at K_HI): an average
        // over the small-tile cutoff takes BM=64, under it BM=32; n % 128 == 0 takes BN=128.
        let big = [50u32, 40, 45, 35, 30];
        let small = [0u32, 0, 4, 3, 2];
        for dt in [DType::Q4K, DType::Q6K] {
            let tag = if dt == DType::Q4K { "q4k" } else { "q6k" };
            for (counts, n, tile) in [
                (&big, 192, "xp"),
                (&big, 256, "xp128"),
                (&small, 192, "xp32"),
                (&small, 256, "xp32w"),
            ] {
                expert_gemm_hi_case(
                    &be,
                    dt,
                    counts,
                    n,
                    &format!("native_gemm_mmq_{tag}_{tile}_hi"),
                );
            }
        }
    }

    fn expert_gemm_hi_case(
        be: &VulkanBackend,
        dt: DType,
        counts_host: &[u32],
        n: usize,
        kernel: &str,
    ) {
        let k = K_HI;
        let nblk = k / 32;
        let n_expert = counts_host.len();
        let stride = k * n;
        let banks: Vec<Vec<u8>> = (0..n_expert)
            .map(|e| synth_bank(dt, stride, 0x41c7 ^ ((e as u64) << 32) ^ n as u64))
            .collect();
        let w_host: Vec<Vec<f32>> = banks
            .iter()
            .map(|b| infr_gguf::dequant::dequant_block(dt, b).unwrap())
            .collect();
        let mins: Vec<Vec<f32>> = banks.iter().map(|b| block_mins(dt, b)).collect();
        let bank_bytes = banks.concat();
        let flat = crate::linear::pad_to_u32_align(&bank_bytes);
        let w = be.alloc(flat.len(), BufferUsage::Weights).unwrap();
        be.upload(w.as_ref(), &flat).unwrap();
        let offsets_host: Vec<u32> = counts_host
            .iter()
            .scan(0u32, |acc, &c| {
                let o = *acc;
                *acc += c;
                Some(o)
            })
            .collect();
        let rows = counts_host.iter().sum::<u32>() as usize;
        // The GEMM's activation stage reads up to a tile past a segment's end.
        let npad = rows.div_ceil(64) * 64 + 64;
        let x = hi_rows(rows);
        let xbuf = upload_f32(be, &x);
        let (plain, hi) = (Quantized::new(be, npad, k), Quantized::new(be, npad, k));
        let out = || be.alloc(npad * n * 4, BufferUsage::Activations).unwrap();
        let (out_plain, out_hi) = (out(), out());
        let counts = upload_u32(be, counts_host);
        let offsets = upload_u32(be, &offsets_host);
        let with_sact = infr_core::tensor::moe_mmq_needs_sact(dt);
        let rec = be.recorder().unwrap();
        for (qz, hres, c) in [
            (&plain, None, &out_plain),
            (&hi, Some(hi.hres.as_ref()), &out_hi),
        ] {
            rec.quant_q8_with(
                xbuf.as_ref(),
                qz.qa.as_ref(),
                qz.dact.as_ref(),
                qz.sact.as_ref(),
                hres,
                rows,
                k,
            );
            rec.matmul_mmq_experts_with(
                dt,
                "expert_gateup",
                qz.qa.as_ref(),
                qz.dact.as_ref(),
                with_sact.then(|| qz.sact.as_ref()),
                hres,
                w.as_ref(),
                0,
                stride,
                counts.as_ref(),
                offsets.as_ref(),
                c.as_ref(),
                rows,
                k,
                n,
                n_expert,
                1,
            );
        }
        rec.finish().unwrap();
        assert!(
            be.built_kernel_names().contains(&kernel),
            "{dt:?}: {kernel} was not built"
        );
        let dl = |b: &dyn Buffer| {
            let mut v = vec![0f32; npad * n];
            be.download(b, bytemuck::cast_slice_mut(&mut v)).unwrap();
            v
        };
        let (y_plain, y_hi) = (dl(out_plain.as_ref()), dl(out_hi.as_ref()));
        let [qa_h, da_h, sa_h, h_h] = hi.host(be);
        let (mut worst, mut outlier_rows, mut gain) = (0f64, 0, f64::INFINITY);
        // Byte positions (lane % 4) of the remainders the routed rows carried.
        let mut lanes_seen = [false; 4];
        for (e, (&off, &cnt)) in offsets_host.iter().zip(counts_host).enumerate() {
            for r in off as usize..(off + cnt) as usize {
                let has_split =
                    (0..nblk).any(|b| hi_split(&x[r * k + b * 32..r * k + (b + 1) * 32]).1);
                // The rows `hi_rows` gives a massive activation (rows 0 and 2 of every four).
                let massive = x[r * k..(r + 1) * k].iter().any(|v| v.abs() >= 50.0);
                let (mut err_hi, mut err_plain) = (0f64, 0f64);
                for o in 0..n {
                    let wr = &w_host[e][o * k..(o + 1) * k];
                    let (mut want, mut mag) = (0f64, 0f64);
                    for b in 0..nblk {
                        let i = r * nblk + b;
                        // A split block's scale carries the mark in its sign bit.
                        let (d, s) = (f16_at(&da_h, i).abs(), f16_at(&sa_h, i));
                        let min = mins[e][o * nblk + b] as f64;
                        let mut dot = 0f64;
                        for l in b * 32..(b + 1) * 32 {
                            let t = (qa_h[r * k + l] as i8 as f64) * (wr[l] as f64 - min);
                            dot += t;
                            mag += (t * d).abs();
                        }
                        // `hres` is written for split blocks only (the rule, not the mark).
                        let split = hi_split(&x[r * k + b * 32..r * k + (b + 1) * 32]).1;
                        let (rem, lane) = if split { vec2_at(&h_h, i) } else { (0.0, 0.0) };
                        if rem != 0.0 {
                            lanes_seen[lane as usize % 4] = true;
                        }
                        let carried = rem as f64 * wr[b * 32 + lane as usize] as f64;
                        want += d * dot + s * min + carried;
                        mag += (s * min).abs() + carried.abs();
                    }
                    let g = y_hi[r * n + o] as f64;
                    let err = (g - want).abs();
                    worst = worst.max(err / mag);
                    assert!(
                        err <= 1e-5 * mag,
                        "{kernel}: expert {e} row {r} col {o}: got {g} want {want} (terms {mag})"
                    );
                    let exact: f64 = wr
                        .iter()
                        .zip(&x[r * k..(r + 1) * k])
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum();
                    err_hi += (y_hi[r * n + o] as f64 - exact).powi(2);
                    err_plain += (y_plain[r * n + o] as f64 - exact).powi(2);
                    if !has_split {
                        let p = y_plain[r * n + o] as f64;
                        assert!(
                            (g - p).abs() <= 1e-5 * mag,
                            "{kernel}: row {r} (no outlier) col {o}: {g} against plain {p}"
                        );
                    }
                }
                if has_split {
                    eprintln!(
                        "  {kernel} row {r}: error against the exact dot {:.3e}, plain {:.3e}",
                        err_hi.sqrt(),
                        err_plain.sqrt()
                    );
                }
                if massive {
                    outlier_rows += 1;
                    let g = (err_plain / err_hi).sqrt();
                    gain = gain.min(g);
                    assert!(
                        g >= 10.0,
                        "{kernel}: row {r}: error against the exact dot {:.3e}, plain {:.3e}",
                        err_hi.sqrt(),
                        err_plain.sqrt()
                    );
                }
            }
        }
        assert!(
            outlier_rows > 0,
            "{kernel}: no routed row carried an outlier"
        );
        assert_eq!(
            lanes_seen, [true; 4],
            "{kernel}: the carried lanes miss a byte position of the packed word"
        );
        eprintln!(
            "{kernel}: worst error / term sum {worst:e}; {outlier_rows} outlier rows, each at \
             least {gain:.0}x closer to the exact dot than the plain GEMM"
        );
    }
}
