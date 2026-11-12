//! Pure selection logic the executor calls, free of Metal types so it also builds and is tested
//! off macOS: `tests/policy.rs` includes this file by path, which is how a Linux run checks it.

/// The key entries per lane (`KPL`) the DeltaNet kernels are instantiated for, in the order of
/// the executor's per-variant kernel name lists. The 32 lanes of a simdgroup split a key head, so
/// a head runs only on the instantiation whose `KPL * 32` is exactly its width.
pub(crate) const DELTANET_KPL: [usize; 4] = [1, 2, 4, 8];

/// Index into [`DELTANET_KPL`] of the DeltaNet kernel instantiation for key-head width `kd`, or
/// `None` when no instantiation fits it and the op must take the host path. A wider instantiation
/// is never a fallback: its lanes would walk `lane * KPL + j` past `kd`, into the next head's
/// state.
pub(crate) fn deltanet_kpl_index(kd: usize) -> Option<usize> {
    DELTANET_KPL.iter().position(|&kpl| kpl * 32 == kd)
}

/// The two f16 flash prefill kernel shapes, which read the KV cache in different block sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FlashKernel {
    /// `attnflash_f16kv`: one simdgroup per 8-query tile, two 8-row score fragments per step.
    Single,
    /// `attnflash2_f16kv_*` / `attnflash2_c128_*`: cooperative, 8-row KV fragments.
    Coop,
}

/// KV rows a causal f16 flash prefill kernel reads for `rows` queries starting at position `pos`
/// — exactly, from the kernels' loops. The extra rows past the newest key get P = 0, but
/// `0 × NaN` is NaN, so they must exist in the bound buffer and hold finite values.
///
/// - `Single`: each full 8-query tile loads 16-row blocks from a 16-aligned start while the block
///   start is at or before the tile's newest key, so the last full tile (newest key
///   `pos + floor8(rows) - 1`) reads to the 16-row boundary past it. A partial last tile runs a
///   serial per-query loop that stops at its own newest key, `pos + rows - 1`.
/// - `Coop`: 8-row fragments from a block-aligned start, skipped once they begin past the
///   newest key, so reads end at the 8-row boundary past `pos + rows - 1`.
pub(crate) fn flash_kv_rows_read(kernel: FlashKernel, pos: usize, rows: usize) -> usize {
    match kernel {
        FlashKernel::Single => {
            let full = rows / 8 * 8;
            let tiles = if full > 0 {
                (pos + full).next_multiple_of(16)
            } else {
                0
            };
            tiles.max(pos + rows)
        }
        FlashKernel::Coop => (pos + rows).next_multiple_of(8),
    }
}

/// Can `kernel` run `rows` queries at `pos` over K/V buffers holding `cap_rows` rows?
pub(crate) fn flash_kv_reads_fit(
    kernel: FlashKernel,
    pos: usize,
    rows: usize,
    cap_rows: usize,
) -> bool {
    flash_kv_rows_read(kernel, pos, rows) <= cap_rows
}

/// The multiple Metal's KV caches (and the f16 dequant scratch) round their row capacity up to:
/// the largest flash block, so a cache holding the context rounded up to it holds every row
/// either flash kernel reads for any prefill that fits the context.
pub(crate) const KV_ROW_ALIGN: usize = 16;

/// The attention kernel families the executor chooses between for a cache read as f16 (natively
/// or through the dequant scratch) or f32. The q8, native-q4 and canvas paths are decided before
/// this; the executor maps each family to its kernel name for the head size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttnKernel {
    /// Cooperative flash at 128 KV positions per step (the 4-row, many-head, hd 128 shape).
    Flash2C128,
    /// Cooperative flash, per head-size instantiation (hd 64/128/256).
    Flash2,
    /// Single-simdgroup flash (any hd % 8 == 0 up to 128).
    Flash,
    /// Vector flash for long-context decode (hd 64/128/256).
    Vec,
    /// 32-way split-KV.
    Split32,
    /// 8-way split-KV.
    Split,
    /// One simdgroup per (query, head).
    Plain,
}

/// A pipeline [`select_attention`] asks the device about: does it exist and take the threadgroup
/// width its kernel needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttnPipe {
    Flash2C128,
    Flash2,
    Vec,
    Split32,
    Split,
}

/// The launch shape [`select_attention`] routes on.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AttnShape {
    /// The kernels read f16 K/V (an f16 cache or the dequant scratch); f32 otherwise.
    pub(crate) f16: bool,
    pub(crate) rows: usize,
    pub(crate) n_head: usize,
    pub(crate) kv_len: usize,
    pub(crate) head_dim: usize,
    /// Position of the first query row.
    pub(crate) pos: usize,
    /// Rows the bound K and V buffers both hold.
    pub(crate) kv_cap_rows: usize,
}

/// The attention kernel family for `s`. `usable` answers whether a pipeline can run; it is asked
/// in routing order. The four-row cooperative pipeline is asked about only once its shape
/// conditions hold; the standard cooperative one is asked about for every head size it is
/// instantiated for, whatever the launch.
///
/// Wide launches on an f16 cache take flash: cooperative where its head size is instantiated and
/// its pipeline fits, else single-simdgroup (which tops out at hd 128) — each only when the bound
/// buffers hold every row it reads ([`flash_kv_reads_fit`]). Otherwise the split-KV family serves
/// short-launch or long-context shapes (vector flash first on f16, then 32-way, then 8-way, each
/// gated on its pipeline) and the plain kernel the rest.
pub(crate) fn select_attention<E>(
    s: &AttnShape,
    mut usable: impl FnMut(AttnPipe) -> Result<bool, E>,
) -> Result<AttnKernel, E> {
    let (rows, nh, kv_len, hd) = (s.rows, s.n_head, s.kv_len, s.head_dim);
    let fits = |k| flash_kv_reads_fit(k, s.pos, rows, s.kv_cap_rows);
    let instantiated = matches!(hd, 64 | 128 | 256);
    let flash2 = if s.f16
        && rows == 4
        && nh >= 16
        && kv_len >= 64
        && hd == 128
        && usable(AttnPipe::Flash2C128)?
    {
        Some(AttnKernel::Flash2C128)
    } else if instantiated && usable(AttnPipe::Flash2)? {
        Some(AttnKernel::Flash2)
    } else {
        None
    };
    let flash2 = flash2.filter(|_| fits(FlashKernel::Coop));
    let launch_wide = rows * nh >= 128;
    let launch_four = rows == 4 && rows * nh >= 64 && flash2.is_some();
    let prefer_flash = s.f16
        && (launch_wide || launch_four)
        && kv_len >= 64
        && hd.is_multiple_of(8)
        && (hd <= 128 || flash2.is_some());
    if prefer_flash {
        if let Some(k) = flash2 {
            return Ok(k);
        }
        if fits(FlashKernel::Single) {
            return Ok(AttnKernel::Flash);
        }
    }
    let split = rows * nh < 128 || kv_len >= 128;
    if s.f16 && split && kv_len >= 128 && instantiated && usable(AttnPipe::Vec)? {
        return Ok(AttnKernel::Vec);
    }
    if split && kv_len >= 128 && hd <= 128 && usable(AttnPipe::Split32)? {
        return Ok(AttnKernel::Split32);
    }
    if split && usable(AttnPipe::Split)? {
        return Ok(AttnKernel::Split);
    }
    Ok(AttnKernel::Plain)
}

/// The capacity, in elements, to allocate for a pooled scratch buffer that holds `have` elements
/// (`None`: nothing pooled yet) when an op needs `need`, on a device whose largest buffer holds
/// `max` elements; `None` means the pooled buffer already fits and is reused. Growth at least
/// doubles, so a request that grows by a little per call (the KV dequant prepass grows by one row
/// per decode token) reallocates a logarithmic number of times over a generation rather than on
/// every token; a first request, or a jump past double, gets exactly what it asked for. Doubling
/// never takes a request that fits the device past `max`; one that does not fit is asked for as
/// is, so the allocation error names its size.
pub(crate) fn scratch_capacity(have: Option<usize>, need: usize, max: usize) -> Option<usize> {
    match have {
        Some(cap) if cap >= need => None,
        Some(cap) => {
            let grown = need.max(cap.saturating_mul(2));
            Some(if need <= max { grown.min(max) } else { grown })
        }
        None => Some(need),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deltanet_picks_only_an_exact_lane_width() {
        for (i, kpl) in DELTANET_KPL.iter().enumerate() {
            assert_eq!(deltanet_kpl_index(kpl * 32), Some(i));
        }
        // Multiples of 32 between the instantiations: each would have run the 8-per-lane kernel.
        for kd in [96, 160, 192, 224] {
            assert_eq!(deltanet_kpl_index(kd), None, "head_k {kd}");
        }
        for kd in [0, 16, 100, 288, 512] {
            assert_eq!(deltanet_kpl_index(kd), None, "head_k {kd}");
        }
    }

    /// One past the highest KV row `kernel` reads, by walking the shader's own loops (no
    /// sliding window: a window only raises where the loops start). The cooperative kernel is
    /// walked at both of its KV step widths.
    fn simulated_rows_read(kernel: FlashKernel, pos: usize, rows: usize) -> usize {
        let mut end = 0;
        let mut r0 = 0;
        match kernel {
            FlashKernel::Single => {
                // attnflash_f16kv: full tiles step 16-row blocks while `j0 <= abs_max`; a partial
                // tile runs per query up to `abs`.
                while r0 + 8 <= rows {
                    let abs_max = pos + r0 + 7;
                    let mut j0 = 0;
                    while j0 <= abs_max {
                        end = end.max(j0 + 16);
                        j0 += 16;
                    }
                    r0 += 8;
                }
                for ti in r0..rows {
                    end = end.max(pos + ti + 1);
                }
            }
            FlashKernel::Coop => {
                for c in [64, 128] {
                    let mut r0 = 0;
                    while r0 < rows {
                        let abs_max = pos + (rows - 1).min(r0 + 7);
                        let mut ic = 0;
                        while ic <= abs_max {
                            // K: fragments starting at or before the newest key.
                            for f in 0..c / 8 {
                                if ic + 8 * f <= abs_max {
                                    end = end.max(ic + 8 * f + 8);
                                }
                            }
                            // V: `nblk` 8-row blocks.
                            let nblk = (c / 8).min((abs_max - ic) / 8 + 1);
                            end = end.max(ic + 8 * nblk);
                            ic += c;
                        }
                        r0 += 8;
                    }
                }
            }
        }
        end
    }

    /// `flash_kv_rows_read` is the exact read extent: never short of a row the loops touch (that
    /// would admit a launch reading past the buffer), never past one (that would refuse a launch
    /// that fits).
    #[test]
    fn flash_reads_match_the_kernel_loops() {
        for kernel in [FlashKernel::Single, FlashKernel::Coop] {
            for pos in 0..200 {
                for rows in 1..=40 {
                    assert_eq!(
                        flash_kv_rows_read(kernel, pos, rows),
                        simulated_rows_read(kernel, pos, rows),
                        "{kernel:?} pos {pos} rows {rows}"
                    );
                }
            }
        }
        // The `attention_flash_hd72_matches_reference` shape: the single-simdgroup kernel's last
        // full tile ends at key 134 and its 16-row block runs to row 143, past a 136-row cache.
        assert_eq!(flash_kv_rows_read(FlashKernel::Single, 119, 17), 144);
        assert!(!flash_kv_reads_fit(FlashKernel::Single, 119, 17, 136));
        assert!(flash_kv_reads_fit(FlashKernel::Coop, 119, 17, 136));
        // A partial last tile stops at its newest key: 17 rows from 0 read 17, not 32.
        assert!(flash_kv_reads_fit(FlashKernel::Single, 0, 17, 17));
    }

    /// The prefill chunks, `(pos, rows)`, of a `p`-token prompt at chunk height `ubatch`.
    fn chunks(p: usize, ubatch: usize) -> impl Iterator<Item = (usize, usize)> {
        (0..p)
            .step_by(ubatch)
            .map(move |pos| (pos, ubatch.min(p - pos)))
    }

    /// A one-shot run sizes its cache at `prompt + max_new + 1` rows; padded to `KV_ROW_ALIGN`,
    /// every prefill chunk fits both flash kernels whatever the reply length — including the
    /// short replies (TTFT, `-n 0`) where the cache ends within a block of the prompt.
    #[test]
    fn a_padded_cache_holds_every_flash_read() {
        for p in 1usize..=600 {
            for max_new in 0..=16 {
                let cap = (p + max_new + 1).next_multiple_of(KV_ROW_ALIGN);
                for ubatch in [8, 64, 512] {
                    for (pos, rows) in chunks(p, ubatch) {
                        for kernel in [FlashKernel::Single, FlashKernel::Coop] {
                            assert!(
                                flash_kv_reads_fit(kernel, pos, rows, cap),
                                "{kernel:?}: prompt {p}, max_new {max_new}, chunk {pos}+{rows}, \
                                 cache {cap}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn shape(f16: bool, rows: usize, nh: usize, kv_len: usize, hd: usize) -> AttnShape {
        AttnShape {
            f16,
            rows,
            n_head: nh,
            kv_len,
            head_dim: hd,
            pos: kv_len - rows,
            kv_cap_rows: kv_len.next_multiple_of(KV_ROW_ALIGN),
        }
    }

    fn pick(s: AttnShape, usable: impl Fn(AttnPipe) -> bool) -> AttnKernel {
        select_attention(&s, |p| Ok::<bool, ()>(usable(p))).unwrap()
    }

    fn any(_: AttnPipe) -> bool {
        true
    }

    #[test]
    fn wide_f16_prefill_takes_flash_per_head_size() {
        for hd in [64, 128, 256] {
            assert_eq!(
                pick(shape(true, 512, 32, 512, hd), any),
                AttnKernel::Flash2,
                "hd {hd}"
            );
        }
        assert_eq!(pick(shape(true, 512, 32, 512, 72), any), AttnKernel::Flash);
        // A quantised cache reads the f16 dequant scratch, padded the same way.
        assert_eq!(
            pick(shape(true, 100, 32, 300, 128), any),
            AttnKernel::Flash2
        );
        // Without its pipeline the cooperative kernel's head sizes up to 128 fall back to the
        // single-simdgroup one; hd 256 has none and leaves flash.
        let no_flash2 = |p| p != AttnPipe::Flash2 && p != AttnPipe::Flash2C128;
        assert_eq!(
            pick(shape(true, 512, 32, 512, 128), no_flash2),
            AttnKernel::Flash
        );
        assert_eq!(
            pick(shape(true, 512, 32, 512, 256), no_flash2),
            AttnKernel::Vec
        );
    }

    /// The read-fit gate: an exactly-sized buffer refuses the kernel whose blocks run past it.
    #[test]
    fn flash_needs_the_rows_it_reads() {
        let exact = |mut s: AttnShape| {
            s.kv_cap_rows = s.kv_len;
            s
        };
        // hd 72 (single-simdgroup only), 17 rows at pos 119: reads to row 143.
        assert_eq!(pick(shape(true, 17, 8, 136, 72), any), AttnKernel::Flash);
        assert_eq!(
            pick(exact(shape(true, 17, 8, 136, 72)), any),
            AttnKernel::Split32
        );
        // hd 128, 129 rows from 0: the cooperative kernel reads to 136, the single one to 129.
        assert_eq!(pick(shape(true, 129, 8, 129, 128), any), AttnKernel::Flash2);
        assert_eq!(
            pick(exact(shape(true, 129, 8, 129, 128)), any),
            AttnKernel::Flash
        );
        // hd 256 has no single-simdgroup flash: it leaves flash entirely.
        assert_eq!(
            pick(exact(shape(true, 129, 8, 129, 256)), any),
            AttnKernel::Vec
        );
    }

    #[test]
    fn four_row_launches_take_the_c128_flash_only_where_it_runs() {
        assert_eq!(
            pick(shape(true, 4, 16, 4096, 128), any),
            AttnKernel::Flash2C128
        );
        let no_c128 = |p| p != AttnPipe::Flash2C128;
        assert_eq!(
            pick(shape(true, 4, 16, 4096, 128), no_c128),
            AttnKernel::Flash2
        );
        // No cooperative pipeline at all: 64 head-rows is not a wide launch, so no flash.
        let no_flash2 = |p| p != AttnPipe::Flash2 && p != AttnPipe::Flash2C128;
        assert_eq!(
            pick(shape(true, 4, 16, 4096, 128), no_flash2),
            AttnKernel::Vec
        );
        // Eight rows never ask for the C128 pipeline, so a device never builds it for them.
        let asked = std::cell::Cell::new(false);
        let k = pick(shape(true, 8, 16, 4096, 128), |p| {
            asked.set(asked.get() || p == AttnPipe::Flash2C128);
            true
        });
        assert_eq!(k, AttnKernel::Flash2);
        assert!(
            !asked.get(),
            "non-four-row shapes must not build the C128 pipeline"
        );
    }

    #[test]
    fn narrow_short_or_f32_launches_never_take_flash() {
        // Three rows x 16 heads, one row x 64 heads: not wide.
        assert_eq!(pick(shape(true, 3, 16, 4096, 128), any), AttnKernel::Vec);
        assert_eq!(pick(shape(true, 1, 64, 4096, 128), any), AttnKernel::Vec);
        // Short context: the plain kernel (no split either, at a wide launch).
        assert_eq!(pick(shape(true, 32, 16, 63, 128), any), AttnKernel::Plain);
        // f32 cache: the split family by context length.
        assert_eq!(
            pick(shape(false, 512, 32, 512, 128), any),
            AttnKernel::Split32
        );
        assert_eq!(
            pick(shape(false, 512, 32, 512, 256), any),
            AttnKernel::Split
        );
    }

    /// Decode degrades through the split family as pipelines go missing.
    #[test]
    fn decode_degrades_through_the_split_family() {
        let s = shape(true, 1, 32, 4096, 128);
        assert_eq!(pick(s, any), AttnKernel::Vec);
        assert_eq!(pick(s, |p| p != AttnPipe::Vec), AttnKernel::Split32);
        assert_eq!(
            pick(s, |p| p != AttnPipe::Vec && p != AttnPipe::Split32),
            AttnKernel::Split
        );
        assert_eq!(pick(s, |_| false), AttnKernel::Plain);
    }

    /// With the cache padded as `kv_row_align` asks, an ordinary one-shot prefill — one chunk of
    /// the whole prompt, a cache of `prompt + max_new + 1` rows — runs on flash for any reply
    /// length, at every head size.
    #[test]
    fn an_ordinary_prefill_runs_on_flash_with_the_padding() {
        for p in 64usize..=300 {
            for max_new in 0..=16 {
                for (hd, want) in [
                    (64, AttnKernel::Flash2),
                    (128, AttnKernel::Flash2),
                    (256, AttnKernel::Flash2),
                    (72, AttnKernel::Flash),
                ] {
                    let s = AttnShape {
                        f16: true,
                        rows: p,
                        n_head: 32,
                        kv_len: p,
                        head_dim: hd,
                        pos: 0,
                        kv_cap_rows: (p + max_new + 1).next_multiple_of(KV_ROW_ALIGN),
                    };
                    assert_eq!(pick(s, any), want, "prompt {p}, max_new {max_new}, hd {hd}");
                }
            }
        }
    }

    /// A device limit no test request comes near.
    const ROOMY: usize = usize::MAX;

    #[test]
    fn scratch_reuses_a_buffer_that_fits() {
        assert_eq!(scratch_capacity(None, 10, ROOMY), Some(10));
        assert_eq!(scratch_capacity(Some(10), 10, ROOMY), None);
        assert_eq!(scratch_capacity(Some(10), 3, ROOMY), None);
    }

    #[test]
    fn scratch_growth_doubles_or_takes_the_jump() {
        assert_eq!(scratch_capacity(Some(10), 11, ROOMY), Some(20));
        assert_eq!(scratch_capacity(Some(10), 50, ROOMY), Some(50));
        assert_eq!(
            scratch_capacity(Some(usize::MAX - 1), usize::MAX, ROOMY),
            Some(usize::MAX)
        );
    }

    /// A tag holding 55% of the device maximum that is asked for 60%: doubling would ask for 110%
    /// and fail where the request itself fits, so the growth stops at the maximum.
    #[test]
    fn scratch_growth_stops_at_the_device_maximum() {
        let max = 1000;
        assert_eq!(scratch_capacity(Some(550), 600, max), Some(max));
        assert_eq!(scratch_capacity(Some(550), max, max), Some(max));
        // A request past the maximum is passed through, for the allocation to refuse by size.
        assert_eq!(scratch_capacity(Some(550), 1200, max), Some(1200));
        assert_eq!(scratch_capacity(None, 1200, max), Some(1200));
    }

    /// The B126 shape: one tag asked for a size that grows by one row per decode token. The pool
    /// must reallocate a logarithmic number of times, and end within double the largest request.
    #[test]
    fn a_per_token_growing_request_reallocates_logarithmically() {
        let row = 8 * 128;
        let mut cap = None;
        let mut allocs = 0;
        for kv_len in 1..=4096 {
            if let Some(c) = scratch_capacity(cap, kv_len * row, ROOMY) {
                cap = Some(c);
                allocs += 1;
            }
        }
        // One per doubling of kv_len from 1, plus the first allocation.
        assert_eq!(allocs, 4096usize.ilog2() as usize + 1);
        let cap = cap.unwrap();
        assert!(cap >= 4096 * row && cap < 2 * 4096 * row);
    }
}
