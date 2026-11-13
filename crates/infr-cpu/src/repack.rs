//! Interleaved 8-row weight packs (Q4_K, Q6_K) for the VNNI GEMM path, and the GEMM kernels
//! that consume them — the (layer, expert) repack-cache payload.
#[cfg(target_arch = "x86_64")]
use crate::pool::SendPtr;
#[cfg(target_arch = "x86_64")]
use crate::quant::Q8;
use infr_core::tensor::DType;
#[cfg(target_arch = "x86_64")]
use infr_gguf::dequant::{k4, rdf16};
use std::collections::HashMap;
#[cfg(target_arch = "x86_64")]
use std::marker::PhantomData;
use std::sync::Arc;

/// Identity of one cached pack: `(buffer uid, byte offset within that buffer, byte length)`.
///
/// The uid — not the slice's ADDRESS — is what makes this key sound. A pack cache outlives the
/// weights it was built from (the same `CpuBackend` survives a serve model reload), and both an
/// allocator and an mmap hand the same address out again afterwards, so an address key returns a
/// pack built from a different weight of equal length: plausible numbers, no error. See
/// `CpuBuffer::uid`.
pub(crate) type PackKey = (u64, usize, usize);

/// `(entries keyed by [`PackKey`], total cached bytes)` — see `repack_cache`'s doc.
pub(crate) type RepackCacheState = (HashMap<PackKey, Arc<Q4kPack>>, usize);

/// Insert `pack` under `key` if still absent, then return the *resident* `Arc` — the caller holds
/// the cache lock. This closes the check-then-insert race in `q4k_pack_for`: the lock drops
/// between the fast-path miss and the (expensive) build, so a second thread may have inserted
/// the same key meanwhile. Re-checking under the lock means only the first insertion counts the
/// bytes (against `budget`) and only one entry lands — the loser returns the winner's pack and drops
/// its own build. Byte accounting happens ONLY on a genuine insert.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn cache_insert_if_absent<P>(
    state: &mut (HashMap<PackKey, Arc<P>>, usize),
    key: PackKey,
    pack: Arc<P>,
    bytes: usize,
    budget: usize,
) -> Arc<P> {
    if let Some(p) = state.0.get(&key) {
        return p.clone();
    }
    if state.1 + bytes <= budget {
        state.1 += bytes;
        state.0.insert(key, pack.clone());
    }
    pack
}

// Only the x86 ilv kernels read these — plain data everywhere else (aarch64 CI builds with
// -D warnings, so the not-x86 dead-code must be explicitly allowed rather than cfg'd away:
// the types appear in cross-target signatures like `expert_matvec_batch`).
/// The interleaved-x8 Q4_K layout (AVX512-VNNI) is the ggml `block_q4_Kx8` idea applied per call:
/// the 8 weight rows' expanded nibbles are INTERLEAVED per (super-block, sub-block, 8-byte group)
/// so one contiguous zmm load carries the SAME 8 activation positions of ALL 8 rows (qword lane i
/// = row i), and the activation qword is broadcast once. Per sub-block that's 4 dpbusd + ONE
/// hadd + ONE mullo for all 8 rows — vs the flat batch8's 8 dpbusd + 16 mullo + per-row hadds —
/// and every weight byte the inner loop touches is sequential.
///
/// `_mm256_hadd_epi32(lo, hi)` yields per-row sums in the fixed permutation [0,1,4,5,2,3,6,7]
/// (its 128-bit-lane semantics); scales are pre-permuted to match and lanes un-permute only at
/// the per-super-block extraction. Bit-identical to the scalar single-row kernel: the integer
/// sums are exact regardless of association, and the per-(row, super-block) f32 expression and
/// its accumulation order are unchanged.
///
/// One 8-output-row group of a Q4_K weight bank in that form: `ilv` is the byte-interleaved
/// nibble expansion, the metadata arrays are PERM-lane-ordered per (super-block, sub-block) so
/// the GEMM can `loadu` them straight into vectors. Cacheable: building this is the per-call
/// repack cost the (layer, expert) cache eliminates (ggml pays it once at load via its
/// `block_q4_Kx8` buffers; we pay it once per cached expert).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[derive(Default)]
pub(crate) struct Q4kPackGroup {
    ilv: Vec<u8>, // [nb * 2048]
    // Pair-duplicated sub-block scales in dpbusd lane order (lanes 2i, 2i+1 = row i): the GEMM
    // scales each sub-block's 16-lane accumulator with ONE 512-bit mullo and adds vertically,
    // deferring the pair-merge to once per SUPER-block (the hadd then lands in PERM order
    // exactly as before). [nb * 8 subs * 16 lanes].
    sc16: Vec<i32>,
    msc: Vec<i32>,  // [nb * 8 subs * 8 lanes], PERM order
    d: Vec<f32>,    // [nb * 8 lanes], PERM order
    dmin: Vec<f32>, // [nb * 8 lanes], PERM order
}

/// A whole Q4_K weight bank (e.g. one MoE expert's fused gate_up) packed as full 8-row groups —
/// `out_f % 8` tail rows are NOT packed (callers run them through the per-row kernel).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) struct Q4kPack {
    pub(crate) groups: Vec<Q4kPackGroup>,
    pub(crate) nb: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Q4kPack {
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) fn bytes(&self) -> usize {
        self.groups
            .iter()
            .map(|g| {
                g.ilv.len() + (g.sc16.len() + g.msc.len()) * 4 + (g.d.len() + g.dmin.len()) * 4
            })
            .sum()
    }
}

/// Build the pack for `wbytes` (a `[out_f, in_f]` Q4_K bank). AVX2 is enough for the expansion.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn q4k_pack(wbytes: &[u8], in_f: usize, out_f: usize) -> Q4kPack {
    let nb = in_f / 256;
    let bpr = wbytes.len() / out_f;
    let groups = (0..out_f / 8)
        .map(|g| {
            let mut pg = Q4kPackGroup::default();
            let rows = std::array::from_fn(|i| &wbytes[(g * 8 + i) * bpr..(g * 8 + i + 1) * bpr]);
            q4k_pack_group_into(&mut pg, DType::Q4K, rows, nb);
            pg
        })
        .collect();
    Q4kPack { groups, nb }
}

/// Fill `pg` with the pack of eight Q4_K or Q5_K (`dt`) weight rows of `nb` super-blocks each,
/// reusing its allocations: the prefill paths pack each 8-row group per call into a per-thread
/// scratch group instead of caching the whole weight. Q5_K shares Q4_K's scale/min layout and
/// f32 expression and differs only in its codes (0..31, the fifth bit from `qh`), which
/// `dpbusd` takes as they are — so both run the same GEMM.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn q4k_pack_group_into(
    pg: &mut Q4kPackGroup,
    dt: DType,
    rows: [&[u8]; 8],
    nb: usize,
) {
    use std::arch::x86_64::*;
    const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let mut tmp = [[0u8; 256]; 8];
    let mut sc_rows = [[0u32; 8]; 8];
    let mut m_rows = [[0u32; 8]; 8];
    let mut d_rows = [0f32; 8];
    let mut dmin_rows = [0f32; 8];
    // Every element below is overwritten, so a resize (not a zero fill) is enough.
    pg.ilv.resize(nb * 2048, 0);
    pg.sc16.resize(nb * 128, 0);
    pg.msc.resize(nb * 64, 0);
    pg.d.resize(nb * 8, 0.0);
    pg.dmin.resize(nb * 8, 0.0);
    let bsz = match dt {
        DType::Q4K => 144,
        DType::Q5K => 176,
        _ => panic!("q4k_pack_group_into: {dt:?} is not Q4_K or Q5_K"),
    };
    for b in 0..nb {
        for i in 0..8 {
            let blk = &rows[i][b * bsz..b * bsz + bsz];
            d_rows[i] = rdf16(&blk[0..2]);
            dmin_rows[i] = rdf16(&blk[2..4]);
            let scales = &blk[4..16];
            for s in 0..8usize {
                let (sc, mv) = k4(s, scales);
                sc_rows[i][s] = sc;
                m_rows[i][s] = mv;
            }
            // Codes in element order: chunk k's 32 low nibbles, then its 32 high nibbles.
            let qs = if bsz == 144 {
                &blk[16..144]
            } else {
                &blk[48..176]
            };
            for k in 0..4usize {
                let nibs = _mm256_loadu_si256(qs[k * 32..].as_ptr() as *const __m256i);
                let lo = _mm256_and_si256(nibs, mask_lo);
                let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
                _mm256_storeu_si256(tmp[i][k * 64..].as_mut_ptr() as *mut __m256i, lo);
                _mm256_storeu_si256(tmp[i][k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
            }
            if bsz == 176 {
                // Q5_K's fifth bit: bit 2k of qh[l] for chunk k's low half, bit 2k+1 for its
                // high half (`dequantize_row_q5_K`'s u1/u2 masks).
                // Per byte: (qh & bit) == bit gives 0xFF or 0, masked down to the 0x10 it adds.
                let qh = _mm256_loadu_si256(blk[16..48].as_ptr() as *const __m256i);
                let b16 = _mm256_set1_epi8(0x10);
                for k in 0..4usize {
                    for (half, bit) in [(0usize, 1u8 << (2 * k)), (32, 2u8 << (2 * k))] {
                        let m = _mm256_set1_epi8(bit as i8);
                        let hi =
                            _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, m), m), b16);
                        let dst = tmp[i][k * 64 + half..].as_mut_ptr() as *mut __m256i;
                        _mm256_storeu_si256(dst, _mm256_or_si256(_mm256_loadu_si256(dst), hi));
                    }
                }
            }
        }
        for (j, &row) in PERM.iter().enumerate() {
            pg.d[b * 8 + j] = d_rows[row];
            pg.dmin[b * 8 + j] = dmin_rows[row];
        }
        for st in 0..8usize {
            for (j, &row) in PERM.iter().enumerate() {
                pg.msc[b * 64 + st * 8 + j] = m_rows[row][st] as i32;
            }
            for i in 0..8usize {
                pg.sc16[b * 128 + st * 16 + 2 * i] = sc_rows[i][st] as i32;
                pg.sc16[b * 128 + st * 16 + 2 * i + 1] = sc_rows[i][st] as i32;
            }
            for gg in 0..4usize {
                let dst =
                    &mut pg.ilv[b * 2048 + st * 256 + gg * 64..b * 2048 + st * 256 + gg * 64 + 64];
                for i in 0..8 {
                    dst[i * 8..i * 8 + 8]
                        .copy_from_slice(&tmp[i][st * 32 + gg * 8..st * 32 + gg * 8 + 8]);
                }
            }
        }
    }
}

/// Where a GEMM over 8-row groups stores its results: output row `o`, token `r` lands at
/// `o * o_stride + r * r_stride` of a buffer of `len` floats. The o-major form is a group's own
/// `[8, m]` column block; the row-major form writes straight into a Linear's `[m, out_f]` output,
/// which saves the transpose pass a column block needs afterwards.
///
/// It holds the buffer's `&'a mut` borrow, so the buffer outlives it and nothing else touches
/// the buffer meanwhile. It is `Copy` so a pool job's tasks can each hold one; what keeps
/// those copies from racing is the contract on [`Self::store8`].
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(crate) struct GemmOut<'a> {
    ptr: SendPtr<f32>,
    len: usize,
    o_stride: usize,
    r_stride: usize,
    _buf: PhantomData<&'a mut [f32]>,
}

#[cfg(target_arch = "x86_64")]
impl<'a> GemmOut<'a> {
    /// `buf[o * m + r]` — one group's `[8, m]` block (or a whole o-major output).
    pub(crate) fn o_major(buf: &'a mut [f32], m: usize) -> Self {
        GemmOut {
            ptr: SendPtr::new(buf.as_mut_ptr()),
            len: buf.len(),
            o_stride: m,
            r_stride: 1,
            _buf: PhantomData,
        }
    }

    /// `buf[r * out_f + o]` — a row-major `[m, out_f]` output.
    pub(crate) fn row_major(buf: &'a mut [f32], out_f: usize) -> Self {
        GemmOut {
            ptr: SendPtr::new(buf.as_mut_ptr()),
            len: buf.len(),
            o_stride: 1,
            r_stride: out_f,
            _buf: PhantomData,
        }
    }

    /// Store one token's eight results, given in PERM lane order, at rows `o0..o0 + 8`.
    ///
    /// # Safety
    /// No other copy of this `GemmOut` may write the same `(o, r)` elements concurrently
    /// (callers split work by 8-row group). The bounds are checked here.
    #[inline]
    unsafe fn store8(&self, o0: usize, r: usize, lanes: [f32; 8]) {
        const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
        let last = (o0 + 7) * self.o_stride + r * self.r_stride;
        assert!(
            last < self.len,
            "GemmOut: rows {o0}..+8 of token {r} are out of bounds"
        );
        for (j, &row) in PERM.iter().enumerate() {
            *self
                .ptr
                .get()
                .add((o0 + row) * self.o_stride + r * self.r_stride) = lanes[j];
        }
    }
}

/// Fewest tokens for which the packed GEMM beats the per-row batch kernels, per dtype: below
/// it, packing an 8-row group costs more than it saves. Measured single-threaded on a Ryzen 9
/// 9950X3D (AVX-512 VNNI), 4096-in × 2048-row weights, packed time over the per-row batch
/// kernels' (`vec_dot_q*k_batch`) time, best of 5, three runs: Q6_K m=1 1.02-1.04, m=2
/// 0.98-1.01, m=3 0.91-0.94, m=4 0.74-0.79. Q4_K (0.79 at m=1) and Q5_K (0.75 at m=1, with
/// its fifth bit expanded in AVX2) win at every count.
pub(crate) const Q6K_PACKED_MIN_TOKENS: usize = 3;

/// [`Q6K_PACKED_MIN_TOKENS`] by dtype: the packed GEMM's break-even token count (1 = always).
pub(crate) fn packed_min_tokens(dt: DType) -> usize {
    match dt {
        DType::Q6K => Q6K_PACKED_MIN_TOKENS,
        _ => 1,
    }
}

/// Tokens a GEMM tile carries through one pass over a packed group's weights: each 64-byte
/// weight load then feeds this many independent dpbusd chains instead of one. Defined on every
/// arch because the route tests size their token counts by it.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) const TILE_TOKENS: usize = 4;

/// [`q4k_gemm_group_out`]'s core for `T` tokens at once, returning each token's eight sums in
/// PERM lane order. Per token the integer sums and the f32 sequence are exactly the
/// single-token form's (integer arithmetic is exact in any order), so results are bit-identical
/// to it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[inline]
unsafe fn q4k_gemm_tile<const T: usize>(
    pg: &Q4kPackGroup,
    nb: usize,
    q8s: [&Q8; T],
) -> [[f32; 8]; T] {
    use std::arch::x86_64::*;
    // Raw-pointer inner loops: slice-range indexing here was ~7% of ALL CPU as bounds checks
    // (samply leaf `index.rs`). The offsets are validated once instead — pack layout by
    // construction of `q4k_pack_group_into`, activation layout by `quantize_q8`.
    assert!(pg.ilv.len() >= nb * 2048 && pg.sc16.len() >= nb * 128 && pg.msc.len() >= nb * 64);
    assert!(pg.d.len() >= nb * 8 && pg.dmin.len() >= nb * 8);
    for q8 in &q8s {
        assert!(q8.qs.len() >= nb * 256 && q8.bsums.len() >= nb * 8 && q8.d.len() >= nb);
    }
    let ilv_p = pg.ilv.as_ptr();
    let sc16_p = pg.sc16.as_ptr();
    let msc_p = pg.msc.as_ptr();
    let d_p = pg.d.as_ptr();
    let dmin_p = pg.dmin.as_ptr();
    let qs_p: [*const i8; T] = std::array::from_fn(|t| q8s[t].qs.as_ptr());
    let mut sumf = [_mm256_setzero_ps(); T];
    for b in 0..nb {
        // Vertical accumulation: each sub-block's 16-lane sums are scaled with ONE 512-bit
        // mullo (pair-duplicated scales, see `sc16`) and added vertically; the pair-merge hadd
        // runs ONCE per super-block. Integer-exact (Σ_s sc·(a+b) = Σ_s (sc·a + sc·b)).
        let mut sd = [_mm512_setzero_si512(); T];
        let mut sm = [_mm256_setzero_si256(); T];
        for st in 0..8usize {
            let mut acc = [_mm512_setzero_si512(); T];
            for gg in 0..4usize {
                let w =
                    _mm512_loadu_si512(ilv_p.add(b * 2048 + st * 256 + gg * 64) as *const __m512i);
                for t in 0..T {
                    let a = _mm512_set1_epi64(
                        (qs_p[t].add(b * 256 + st * 32 + gg * 8) as *const i64).read_unaligned(),
                    );
                    acc[t] = _mm512_dpbusd_epi32(acc[t], w, a);
                }
            }
            let sc16 = _mm512_loadu_si512(sc16_p.add(b * 128 + st * 16) as *const __m512i);
            let m_v = _mm256_loadu_si256(msc_p.add(b * 64 + st * 8) as *const __m256i);
            for t in 0..T {
                sd[t] = _mm512_add_epi32(sd[t], _mm512_mullo_epi32(acc[t], sc16));
                let isum = _mm256_set1_epi32(*q8s[t].bsums.as_ptr().add(b * 8 + st));
                sm[t] = _mm256_add_epi32(sm[t], _mm256_mullo_epi32(m_v, isum));
            }
        }
        let d_v = _mm256_loadu_ps(d_p.add(b * 8));
        let dmin_v = _mm256_loadu_ps(dmin_p.add(b * 8));
        for t in 0..T {
            let sd_lo = _mm512_castsi512_si256(sd[t]);
            let sd_hi = _mm512_extracti64x4_epi64::<1>(sd[t]);
            let sd_ymm = _mm256_hadd_epi32(sd_lo, sd_hi); // pair-merge -> PERM order
                                                          // q8.d[b] * (d*sd - dmin*sm), 8 rows at once. No FMA: each mul/sub/add rounds
                                                          // separately, matching the scalar expression (cvtepi32→f32 is exact here).
            let sd_f = _mm256_cvtepi32_ps(sd_ymm);
            let sm_f = _mm256_cvtepi32_ps(sm[t]);
            let x = _mm256_sub_ps(_mm256_mul_ps(d_v, sd_f), _mm256_mul_ps(dmin_v, sm_f));
            sumf[t] = _mm256_add_ps(
                sumf[t],
                _mm256_mul_ps(_mm256_set1_ps(*q8s[t].d.as_ptr().add(b)), x),
            );
        }
    }
    let mut out = [[0f32; 8]; T];
    for t in 0..T {
        _mm256_storeu_ps(out[t].as_mut_ptr(), sumf[t]);
    }
    out
}

/// The GEMM over a packed [`Q4kPackGroup`]: its eight output rows for every token of `q8s`,
/// stored at rows `o0..o0 + 8` of `out`. Bit-identical to the scalar oracles `vec_dot_q4k` and
/// `vec_dot_q5k` (whichever dtype the group was packed from).
///
/// # Safety
/// The CPU must have AVX-512BW and AVX-512 VNNI, and `out` must satisfy [`GemmOut::store8`]'s
/// contract for rows `o0..o0 + 8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn q4k_gemm_group_out(
    pg: &Q4kPackGroup,
    nb: usize,
    q8s: &[Q8],
    out: GemmOut<'_>,
    o0: usize,
) {
    let mut r = 0;
    while r + TILE_TOKENS <= q8s.len() {
        let sums = q4k_gemm_tile::<TILE_TOKENS>(pg, nb, std::array::from_fn(|t| &q8s[r + t]));
        for (t, s) in sums.into_iter().enumerate() {
            out.store8(o0, r + t, s);
        }
        r += TILE_TOKENS;
    }
    for (rr, q8) in q8s.iter().enumerate().skip(r) {
        let [s] = q4k_gemm_tile::<1>(pg, nb, [q8]);
        out.store8(o0, rr, s);
    }
}

thread_local! {
    /// Each pool thread's scratch group for [`q4k_pack_and_gemm_group`]: packing per call into a
    /// reused group keeps the per-call repack free of allocations (this kernel used to allocate a
    /// few dozen buffers per 8-row group, contending on the process heap from every thread).
    #[cfg(target_arch = "x86_64")]
    static Q4K_GROUP_SCRATCH: std::cell::RefCell<Q4kPackGroup> = std::cell::RefCell::default();
}

/// Pack eight Q4_K or Q5_K (`dt`) weight rows into this thread's scratch group and run
/// [`q4k_gemm_group_out`] over them: the uncached form of the packed GEMM, for weights not worth
/// a resident pack (dense Linear weights, split MoE gate/up banks).
///
/// # Safety
/// As [`q4k_gemm_group_out`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn q4k_pack_and_gemm_group(
    dt: DType,
    rows: [&[u8]; 8],
    nb: usize,
    q8s: &[Q8],
    out: GemmOut<'_>,
    o0: usize,
) {
    Q4K_GROUP_SCRATCH.with_borrow_mut(|pg| {
        q4k_pack_group_into(pg, dt, rows, nb);
        q4k_gemm_group_out(pg, nb, q8s, out, o0);
    });
}

/// [`Q4kPackGroup`]'s Q6_K sibling: 16 sub-blocks of 16 (2 qwords each) instead of 8×32, int8
/// scales instead of the 6-bit sc/min pairs, and no min term — the `-32` offset rides the
/// activation `bsums16` at GEMM time (split as `-16·bsum` per pair lane, integer-exact).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[derive(Default)]
pub(crate) struct Q6kPackGroup {
    ilv: Vec<u8>, // [nb * 2048]: [st 0..16][qw 0..2][row-interleaved 64 B], codes BIASED (q+32)
    sc16: Vec<i32>, // [nb * 256]: pair-duplicated int8 scales (lanes 2i, 2i+1 = row i)
    d: Vec<f32>,  // [nb * 8], PERM order
}

/// Fill `pg` with the interleaved pack of eight Q6_K weight rows of `nb` super-blocks each,
/// reusing its allocations — see [`q6k_pack_and_gemm_group`]. AVX2 does the code expansion.
///
/// # Safety
/// The CPU must have AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn q6k_pack_group_into(pg: &mut Q6kPackGroup, rows: [&[u8]; 8], nb: usize) {
    use std::arch::x86_64::*;
    const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
    // Every element below is overwritten, so a resize (not a zero fill) is enough.
    pg.ilv.resize(nb * 2048, 0);
    pg.sc16.resize(nb * 256, 0);
    pg.d.resize(nb * 8, 0.0);
    let mut tmp = [[0u8; 256]; 8];
    let mut sc_rows = [[0i32; 16]; 8];
    let mut d_rows = [0f32; 8];
    for b in 0..nb {
        for i in 0..8 {
            let blk = &rows[i][b * 210..b * 210 + 210];
            let ql = &blk[0..128];
            let qh = &blk[128..192];
            let scales = &blk[192..208];
            d_rows[i] = rdf16(&blk[208..210]);
            for (st, sv) in sc_rows[i].iter_mut().enumerate() {
                *sv = scales[st] as i8 as i32;
            }
            // Linear element order, codes biased (q+32, 0..63) — the same mapping the batch
            // kernels' flat expansion uses. The 16-bit shifts are safe bytewise: every shifted
            // value is masked to bits that came from (or stay inside) its own byte.
            let (m4, m3) = (_mm256_set1_epi8(0x0F), _mm256_set1_epi8(0x03));
            for half in 0..2usize {
                let (qlo, qho, base) = (half * 64, half * 32, half * 128);
                let a = _mm256_loadu_si256(ql[qlo..].as_ptr() as *const __m256i);
                let b = _mm256_loadu_si256(ql[qlo + 32..].as_ptr() as *const __m256i);
                let h = _mm256_loadu_si256(qh[qho..].as_ptr() as *const __m256i);
                let hi2 = |sh: i32| {
                    let v = match sh {
                        0 => h,
                        2 => _mm256_srli_epi16::<2>(h),
                        4 => _mm256_srli_epi16::<4>(h),
                        _ => _mm256_srli_epi16::<6>(h),
                    };
                    _mm256_slli_epi16::<4>(_mm256_and_si256(v, m3))
                };
                let lo4 = |v: __m256i| _mm256_and_si256(v, m4);
                let hi4 = |v: __m256i| _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4);
                let f = tmp[i][base..].as_mut_ptr() as *mut __m256i;
                _mm256_storeu_si256(f, _mm256_or_si256(lo4(a), hi2(0)));
                _mm256_storeu_si256(f.add(1), _mm256_or_si256(lo4(b), hi2(2)));
                _mm256_storeu_si256(f.add(2), _mm256_or_si256(hi4(a), hi2(4)));
                _mm256_storeu_si256(f.add(3), _mm256_or_si256(hi4(b), hi2(6)));
            }
        }
        for (j, &row) in PERM.iter().enumerate() {
            pg.d[b * 8 + j] = d_rows[row];
        }
        for st in 0..16usize {
            for i in 0..8usize {
                pg.sc16[b * 256 + st * 16 + 2 * i] = sc_rows[i][st];
                pg.sc16[b * 256 + st * 16 + 2 * i + 1] = sc_rows[i][st];
            }
            for qw in 0..2usize {
                let dst =
                    &mut pg.ilv[b * 2048 + st * 128 + qw * 64..b * 2048 + st * 128 + qw * 64 + 64];
                for i in 0..8 {
                    dst[i * 8..i * 8 + 8]
                        .copy_from_slice(&tmp[i][st * 16 + qw * 8..st * 16 + qw * 8 + 8]);
                }
            }
        }
    }
}

thread_local! {
    /// Each pool thread's scratch group for [`q6k_pack_and_gemm_group`].
    #[cfg(target_arch = "x86_64")]
    static Q6K_GROUP_SCRATCH: std::cell::RefCell<Q6kPackGroup> = std::cell::RefCell::default();
}

/// Pack eight Q6_K weight rows into this thread's scratch group and run [`q6k_gemm_group_out`]
/// over them — the Q6_K twin of [`q4k_pack_and_gemm_group`].
///
/// # Safety
/// As [`q6k_gemm_group_out`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn q6k_pack_and_gemm_group(
    rows: [&[u8]; 8],
    nb: usize,
    q8s: &[Q8],
    out: GemmOut<'_>,
    o0: usize,
) {
    Q6K_GROUP_SCRATCH.with_borrow_mut(|pg| {
        q6k_pack_group_into(pg, rows, nb);
        q6k_gemm_group_out(pg, nb, q8s, out, o0);
    });
}

/// [`q6k_gemm_group_out`]'s core for `T` tokens at once, returning each token's eight sums in
/// PERM lane order. Same integer core as `vec_dot_q6k_batch_vnni` (dpbusd on biased codes,
/// `-32·bsum16` split as `-16·bsum16` per pair lane — exact) and the same per-super-block f32
/// sequence (`(d · q8.d) · isum`) per token, so results are bit-identical to the batch/single
/// kernels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[inline]
unsafe fn q6k_gemm_tile<const T: usize>(
    pg: &Q6kPackGroup,
    nb: usize,
    q8s: [&Q8; T],
) -> [[f32; 8]; T] {
    use std::arch::x86_64::*;
    assert!(pg.ilv.len() >= nb * 2048 && pg.sc16.len() >= nb * 256 && pg.d.len() >= nb * 8);
    for q8 in &q8s {
        assert!(q8.qs.len() >= nb * 256 && q8.bsums16.len() >= nb * 16 && q8.d.len() >= nb);
    }
    let ilv_p = pg.ilv.as_ptr();
    let sc16_p = pg.sc16.as_ptr();
    let d_p = pg.d.as_ptr();
    let qs_p: [*const i8; T] = std::array::from_fn(|t| q8s[t].qs.as_ptr());
    let mut sumf = [_mm256_setzero_ps(); T];
    for b in 0..nb {
        let mut sd = [_mm512_setzero_si512(); T];
        for st in 0..16usize {
            let mut acc = [_mm512_setzero_si512(); T];
            for qw in 0..2usize {
                let w =
                    _mm512_loadu_si512(ilv_p.add(b * 2048 + st * 128 + qw * 64) as *const __m512i);
                for t in 0..T {
                    let a = _mm512_set1_epi64(
                        (qs_p[t].add(b * 256 + st * 16 + qw * 8) as *const i64).read_unaligned(),
                    );
                    acc[t] = _mm512_dpbusd_epi32(acc[t], w, a);
                }
            }
            let sc16 = _mm512_loadu_si512(sc16_p.add(b * 256 + st * 16) as *const __m512i);
            for t in 0..T {
                // -32·bsum16 per sub-block, split as -16·bsum16 per pair lane (exact:
                // (a1-16bs)+(a2-16bs) = dp-32bs).
                let corr = _mm512_set1_epi32(16 * *q8s[t].bsums16.as_ptr().add(b * 16 + st));
                sd[t] = _mm512_add_epi32(
                    sd[t],
                    _mm512_mullo_epi32(_mm512_sub_epi32(acc[t], corr), sc16),
                );
            }
        }
        let d_v = _mm256_loadu_ps(d_p.add(b * 8));
        for t in 0..T {
            let sd_lo = _mm512_castsi512_si256(sd[t]);
            let sd_hi = _mm512_extracti64x4_epi64::<1>(sd[t]);
            let sd_ymm = _mm256_hadd_epi32(sd_lo, sd_hi); // pair-merge -> PERM order
            let s_f = _mm256_cvtepi32_ps(sd_ymm);
            let x = _mm256_mul_ps(d_v, _mm256_set1_ps(*q8s[t].d.as_ptr().add(b)));
            sumf[t] = _mm256_add_ps(sumf[t], _mm256_mul_ps(x, s_f));
        }
    }
    let mut out = [[0f32; 8]; T];
    for t in 0..T {
        _mm256_storeu_ps(out[t].as_mut_ptr(), sumf[t]);
    }
    out
}

/// The GEMM over a prebuilt [`Q6kPackGroup`]: its eight output rows for every token of `q8s`,
/// stored at rows `o0..o0 + 8` of `out`. Bit-identical to the batch/single Q6_K kernels.
///
/// # Safety
/// The CPU must have AVX-512BW and AVX-512 VNNI, and `out` must satisfy [`GemmOut::store8`]'s
/// contract for rows `o0..o0 + 8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn q6k_gemm_group_out(
    pg: &Q6kPackGroup,
    nb: usize,
    q8s: &[Q8],
    out: GemmOut<'_>,
    o0: usize,
) {
    let mut r = 0;
    while r + TILE_TOKENS <= q8s.len() {
        let sums = q6k_gemm_tile::<TILE_TOKENS>(pg, nb, std::array::from_fn(|t| &q8s[r + t]));
        for (t, s) in sums.into_iter().enumerate() {
            out.store8(o0, r + t, s);
        }
        r += TILE_TOKENS;
    }
    for (rr, q8) in q8s.iter().enumerate().skip(r) {
        let [s] = q6k_gemm_tile::<1>(pg, nb, [q8]);
        out.store8(o0, rr, s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #4: `cache_insert_if_absent` closes the check-then-insert race — the re-check under the lock
    // must (a) insert + count bytes exactly once for a fresh key, (b) NOT re-count or overwrite when
    // the key is already resident (the race loser gets the winner's Arc), and (c) respect the budget.
    #[test]
    fn cache_insert_if_absent_is_idempotent_and_budgeted() {
        let mut state: (HashMap<PackKey, Arc<u32>>, usize) = (HashMap::new(), 0);
        let budget = 100usize;

        // First insert of a fresh key: lands, counts its bytes.
        let win = cache_insert_if_absent(&mut state, (1, 2, 3), Arc::new(42), 10, budget);
        assert_eq!(*win, 42);
        assert_eq!(state.1, 10, "bytes counted once");
        assert_eq!(state.0.len(), 1);

        // Simulated race: a second builder reaches the lock with the SAME key but its own Arc. It
        // must get the RESIDENT Arc back (Arc::ptr_eq the winner) and must NOT double-count bytes.
        let resident = state.0.get(&(1, 2, 3)).unwrap().clone();
        let loser = cache_insert_if_absent(&mut state, (1, 2, 3), Arc::new(999), 10, budget);
        assert!(
            Arc::ptr_eq(&loser, &resident),
            "loser gets the resident pack"
        );
        assert_eq!(state.1, 10, "no double byte accounting on a re-check hit");
        assert_eq!(state.0.len(), 1);

        // A pack that would blow the budget is built-and-returned but NOT inserted/counted.
        let big = cache_insert_if_absent(&mut state, (4, 5, 6), Arc::new(7), 1_000, budget);
        assert_eq!(*big, 7, "over-budget pack still usable, just transient");
        assert_eq!(state.1, 10, "over-budget insert must not count bytes");
        assert_eq!(state.0.len(), 1, "over-budget key not inserted");
    }
}
