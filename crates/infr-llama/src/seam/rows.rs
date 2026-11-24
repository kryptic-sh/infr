//! Where a layer's logical KV rows sit inside its buffer, and the row copies that follow them.
//!
//! A slot's cache keeps logical row `r` at one of three physical rows: `r` for a linear cache,
//! `r % rows` for a sliding-window ring, or inside the page the slot's table maps `r` to. Every
//! store, restore and page copy walks [`RowLayout::spans`] instead of re-deriving that mapping.
//! A row's byte offset is `kv_fmt_bytes(fmt, physical_row * row_elems)`, except on a backend whose
//! Q8_0 side is planar ([`SideLayout`]): there the codes and the scales are two separate runs.

use std::ops::Range;

use anyhow::{ensure, Result};
use infr_core::backend::{Backend, Buffer};
use infr_core::budget::kv_fmt_bytes;
use infr_core::DType;
use infr_vulkan::paged::PAGE_ROWS;

use super::pool::SlotMap;

/// One contiguous run of logical rows and the physical row it starts at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowSpan {
    pub(crate) logical: usize,
    pub(crate) physical: usize,
    pub(crate) rows: usize,
}

/// Where one layer's logical KV rows live inside its buffer.
pub(crate) trait RowLayout {
    /// The physical runs holding logical rows `start..start + len`, in logical order.
    fn spans(&self, start: usize, len: usize) -> Vec<RowSpan>;

    /// The oldest logical row still resident after `written` rows went in.
    fn oldest_resident(&self, written: usize) -> usize {
        let _ = written;
        0
    }
}

impl<T: RowLayout + ?Sized> RowLayout for &T {
    fn spans(&self, start: usize, len: usize) -> Vec<RowSpan> {
        (**self).spans(start, len)
    }

    fn oldest_resident(&self, written: usize) -> usize {
        (**self).oldest_resident(written)
    }
}

/// Logical row `r` at physical row `r`.
pub(crate) struct Linear;

impl RowLayout for Linear {
    fn spans(&self, start: usize, len: usize) -> Vec<RowSpan> {
        if len == 0 {
            return Vec::new();
        }
        vec![RowSpan {
            logical: start,
            physical: start,
            rows: len,
        }]
    }
}

/// A sliding-window ring of `rows` rows: logical row `r` at physical row `r % rows`.
pub(crate) struct Ring {
    pub(crate) rows: usize,
}

impl RowLayout for Ring {
    fn spans(&self, start: usize, len: usize) -> Vec<RowSpan> {
        let mut out = Vec::new();
        if self.rows == 0 {
            return out;
        }
        let end = start + len;
        let mut r = start;
        while r < end {
            let physical = r % self.rows;
            let rows = (end - r).min(self.rows - physical);
            out.push(RowSpan {
                logical: r,
                physical,
                rows,
            });
            r += rows;
        }
        out
    }

    fn oldest_resident(&self, written: usize) -> usize {
        written.saturating_sub(self.rows)
    }
}

impl RowLayout for SlotMap {
    fn spans(&self, start: usize, len: usize) -> Vec<RowSpan> {
        let end = start + len;
        let mut out = Vec::new();
        let mut r = start;
        while r < end {
            let off = r % PAGE_ROWS;
            let rows = (end - r).min(PAGE_ROWS - off);
            out.push(RowSpan {
                logical: r,
                physical: self.page(r / PAGE_ROWS) as usize * PAGE_ROWS + off,
                rows,
            });
            r += rows;
        }
        out
    }
}

/// Bytes in one row of `row_elems` elements, refused when rows of `fmt` do not pack back to back
/// (a Q8_0 row of an odd block count pads, so span offsets would not add up).
pub(crate) fn row_bytes(fmt: DType, row_elems: usize) -> Result<usize> {
    let one = kv_fmt_bytes(fmt, row_elems);
    ensure!(
        kv_fmt_bytes(fmt, row_elems * 128) == one * 128,
        "a {fmt:?} row of {row_elems} elements does not pack to a whole byte count"
    );
    Ok(one)
}

/// Bytes of one GGUF Q8_0 block, and the elements it holds: a 2-byte f16 scale, then 32 int8 codes.
const Q8_BLOCK_BYTES: usize = 34;
pub(crate) const Q8_BLOCK_ELEMS: usize = 32;

/// How one side's rows sit in its buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SideLayout {
    /// GGUF blocks back to back: a row's bytes are its whole row.
    Packed,
    /// Vulkan's planar Q8_0: every int8 code, then every block's scale from `cap_elems` bytes on.
    PlanarQ8 { cap_elems: usize },
}

impl SideLayout {
    /// The side's layout, from the backend that owns it and the buffer it lives in. A planar side
    /// is refused when its length is not exactly a capacity plus its scales, so a padded or
    /// half-allocated side is a miss rather than a read of somebody else's bytes.
    pub(crate) fn of(be: &dyn Backend, fmt: DType, buf: &dyn Buffer) -> Result<Self> {
        if fmt != DType::Q8_0 || !be.kv_q8_planar() {
            return Ok(Self::Packed);
        }
        let bytes = buf.len_bytes();
        let cap_elems = bytes * Q8_BLOCK_ELEMS / Q8_BLOCK_BYTES;
        ensure!(
            cap_elems.is_multiple_of(Q8_BLOCK_ELEMS)
                && bytes == cap_elems + cap_elems / Q8_BLOCK_ELEMS * 2,
            "a planar Q8_0 side of {bytes} bytes is not {cap_elems} codes plus their scales"
        );
        Ok(Self::PlanarQ8 { cap_elems })
    }
}

/// Gather a planar side's codes and scales into GGUF Q8_0 blocks, one per 32-element block.
fn planar_to_packed(codes: &[u8], scales: &[u8], row_elems: usize, out: &mut [u8]) -> Result<()> {
    ensure!(
        row_elems.is_multiple_of(Q8_BLOCK_ELEMS),
        "a planar Q8_0 row of {row_elems} elements is not whole blocks"
    );
    let blocks = out.len() / Q8_BLOCK_BYTES;
    ensure!(
        codes.len() == blocks * Q8_BLOCK_ELEMS && scales.len() == blocks * 2,
        "planar Q8_0 codes/scales do not cover {blocks} blocks"
    );
    for (b, block) in out
        .as_chunks_mut::<Q8_BLOCK_BYTES>()
        .0
        .iter_mut()
        .enumerate()
    {
        block[..2].copy_from_slice(&scales[b * 2..b * 2 + 2]);
        block[2..].copy_from_slice(&codes[b * Q8_BLOCK_ELEMS..(b + 1) * Q8_BLOCK_ELEMS]);
    }
    Ok(())
}

/// Split GGUF Q8_0 blocks into a planar side's codes and scales.
fn packed_to_planar(
    data: &[u8],
    row_elems: usize,
    codes: &mut [u8],
    scales: &mut [u8],
) -> Result<()> {
    ensure!(
        row_elems.is_multiple_of(Q8_BLOCK_ELEMS),
        "a planar Q8_0 row of {row_elems} elements is not whole blocks"
    );
    let blocks = data.len() / Q8_BLOCK_BYTES;
    ensure!(
        codes.len() == blocks * Q8_BLOCK_ELEMS && scales.len() == blocks * 2,
        "planar Q8_0 codes/scales do not cover {blocks} blocks"
    );
    for (b, block) in data.as_chunks::<Q8_BLOCK_BYTES>().0.iter().enumerate() {
        scales[b * 2..b * 2 + 2].copy_from_slice(&block[..2]);
        codes[b * Q8_BLOCK_ELEMS..(b + 1) * Q8_BLOCK_ELEMS].copy_from_slice(&block[2..]);
    }
    Ok(())
}

/// Download logical `rows` of `buf` into `out`, which holds exactly those rows in order.
pub(crate) fn read_rows(
    be: &dyn Backend,
    buf: &dyn Buffer,
    fmt: DType,
    row_elems: usize,
    layout: &dyn RowLayout,
    rows: Range<usize>,
    out: &mut [u8],
) -> Result<()> {
    let len = rows.len();
    let rb = row_bytes(fmt, row_elems)?;
    ensure!(
        out.len() == len * rb,
        "read_rows: {} bytes for {len} rows of {row_elems} elements",
        out.len()
    );
    let side = SideLayout::of(be, fmt, buf)?;
    for s in layout.spans(rows.start, len) {
        let at = (s.logical - rows.start) * rb;
        let slice = &mut out[at..at + s.rows * rb];
        match side {
            SideLayout::Packed => be.download_at(buf, s.physical * rb, slice)?,
            SideLayout::PlanarQ8 { cap_elems } => {
                let (codes, scales) = download_planar(be, buf, cap_elems, row_elems, s)?;
                planar_to_packed(&codes, &scales, row_elems, slice)?;
            }
        }
    }
    Ok(())
}

/// A span's codes and scales, as the planar buffer holds them.
fn download_planar(
    be: &dyn Backend,
    buf: &dyn Buffer,
    cap_elems: usize,
    row_elems: usize,
    s: RowSpan,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let elems = s.rows * row_elems;
    let base = s.physical * row_elems;
    let mut codes = vec![0u8; elems];
    let mut scales = vec![0u8; elems / Q8_BLOCK_ELEMS * 2];
    be.download_at(buf, base, &mut codes)?;
    be.download_at(buf, cap_elems + base / Q8_BLOCK_ELEMS * 2, &mut scales)?;
    Ok((codes, scales))
}

/// Upload `data`, which holds exactly logical `rows` in order, into `buf`.
pub(crate) fn write_rows(
    be: &dyn Backend,
    buf: &dyn Buffer,
    fmt: DType,
    row_elems: usize,
    layout: &dyn RowLayout,
    rows: Range<usize>,
    data: &[u8],
) -> Result<()> {
    let len = rows.len();
    let rb = row_bytes(fmt, row_elems)?;
    ensure!(
        data.len() == len * rb,
        "write_rows: {} bytes for {len} rows of {row_elems} elements",
        data.len()
    );
    let side = SideLayout::of(be, fmt, buf)?;
    for s in layout.spans(rows.start, len) {
        let at = (s.logical - rows.start) * rb;
        let slice = &data[at..at + s.rows * rb];
        match side {
            SideLayout::Packed => be.upload_at(buf, s.physical * rb, slice)?,
            SideLayout::PlanarQ8 { cap_elems } => {
                let elems = s.rows * row_elems;
                let base = s.physical * row_elems;
                let mut codes = vec![0u8; elems];
                let mut scales = vec![0u8; elems / Q8_BLOCK_ELEMS * 2];
                packed_to_planar(slice, row_elems, &mut codes, &mut scales)?;
                be.upload_at(buf, base, &codes)?;
                be.upload_at(buf, cap_elems + base / Q8_BLOCK_ELEMS * 2, &scales)?;
            }
        }
    }
    Ok(())
}

/// Copy logical `rows` from `src` to `dst` on the device, one copy per run both layouts agree on.
/// A planar side moves as two runs, its codes and its scales; a planar run into a packed buffer (or
/// a mixed pair, which no engine builds) goes through host memory.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_rows(
    be: &dyn Backend,
    src: &dyn Buffer,
    src_layout: &dyn RowLayout,
    dst: &dyn Buffer,
    dst_layout: &dyn RowLayout,
    fmt: DType,
    row_elems: usize,
    rows: Range<usize>,
) -> Result<()> {
    if row_elems == 0 {
        return Ok(());
    }
    let rb = row_bytes(fmt, row_elems)?;
    let side = SideLayout::of(be, fmt, src)?;
    let dst_side = SideLayout::of(be, fmt, dst)?;
    if side != dst_side
        && matches!(side, SideLayout::PlanarQ8 { .. })
            != matches!(dst_side, SideLayout::PlanarQ8 { .. })
    {
        let mut host = vec![0u8; rows.len() * rb];
        read_rows(be, src, fmt, row_elems, src_layout, rows.clone(), &mut host)?;
        return write_rows(be, dst, fmt, row_elems, dst_layout, rows, &host);
    }
    let a = src_layout.spans(rows.start, rows.len());
    let b = dst_layout.spans(rows.start, rows.len());
    let same_buffer = std::ptr::addr_eq(src, dst);
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let (x, y) = (a[i], b[j]);
        let lo = x.logical.max(y.logical);
        let hi = (x.logical + x.rows).min(y.logical + y.rows);
        if lo < hi {
            let from = x.physical + (lo - x.logical);
            let to = y.physical + (lo - y.logical);
            let n = (hi - lo) * row_elems;
            match (side, dst_side) {
                (SideLayout::Packed, SideLayout::Packed) => {
                    if !(same_buffer && from == to) {
                        be.copy_buffer_at(src, from * rb, dst, to * rb, (hi - lo) * rb)?;
                    }
                }
                (
                    SideLayout::PlanarQ8 { cap_elems: scap },
                    SideLayout::PlanarQ8 { cap_elems: dcap },
                ) => {
                    let (s_base, d_base) = (from * row_elems, to * row_elems);
                    be.copy_buffer_at(src, s_base, dst, d_base, n)?;
                    let scales = n / Q8_BLOCK_ELEMS * 2;
                    be.copy_buffer_at(
                        src,
                        scap + s_base / Q8_BLOCK_ELEMS * 2,
                        dst,
                        dcap + d_base / Q8_BLOCK_ELEMS * 2,
                        scales,
                    )?;
                }
                _ => unreachable!("a mixed pair went through host memory above"),
            }
        }
        if x.logical + x.rows <= y.logical + y.rows {
            i += 1;
        } else {
            j += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::{KvPool, PagedSlot};
    use crate::Config;
    use infr_core::backend::BufferUsage;

    #[test]
    fn a_padded_q8_row_is_refused_and_an_even_one_packs() {
        assert!(row_bytes(DType::Q8_0, 32).is_err());
        assert_eq!(row_bytes(DType::Q8_0, 128).unwrap(), 136);
        assert!(row_bytes(DType::Turbo2, 64).is_err());
        assert_eq!(row_bytes(DType::F16, 96).unwrap(), 192);
    }

    fn span(logical: usize, physical: usize, rows: usize) -> RowSpan {
        RowSpan {
            logical,
            physical,
            rows,
        }
    }

    fn cfg() -> Config {
        Config {
            n_layer: 2,
            n_head: 4,
            n_kv: 2,
            n_embd: 64,
            head_dim: 16,
            rope_dim: 16,
            ..Default::default()
        }
    }

    /// A pool whose second slot maps logical pages 0, 1, 2 to physical pages 2, 3, 4.
    fn two_slots(be: &dyn Backend) -> (std::sync::Arc<KvPool>, PagedSlot, PagedSlot) {
        let pool = KvPool::new(be, &cfg(), 1536, DType::F16, DType::F16, 5).unwrap();
        let mut a = PagedSlot::new(&pool, be).unwrap();
        a.reserve_rows(be, 1).unwrap();
        let mut b = PagedSlot::new(&pool, be).unwrap();
        b.reserve_rows(be, 1536).unwrap();
        assert_eq!((a.map.page(0), b.map.page(0), b.map.page(2)), (1, 2, 4));
        (pool, a, b)
    }

    #[test]
    fn a_planar_side_interleaves_into_gguf_blocks() {
        // Two rows of 64 elements: four blocks, each a 2-byte scale then 32 codes.
        let row_elems = 64;
        let mut packed = Vec::new();
        for (row, block) in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
            packed.extend_from_slice(&[row, block]);
            packed.extend(std::iter::repeat_n(row * 16 + block, 32));
        }
        let mut codes = vec![0u8; 128];
        let mut scales = vec![0u8; 8];
        packed_to_planar(&packed, row_elems, &mut codes, &mut scales).unwrap();
        assert_eq!(&codes[..32], &[0u8; 32]);
        assert_eq!(&codes[32..64], &[1u8; 32]);
        assert_eq!(&codes[64..96], &[16u8; 32]);
        assert_eq!(&codes[96..], &[17u8; 32]);
        assert_eq!(scales, [0, 0, 0, 1, 1, 0, 1, 1]);

        let mut back = vec![0u8; packed.len()];
        planar_to_packed(&codes, &scales, row_elems, &mut back).unwrap();
        assert_eq!(back, packed);

        // A count that does not match the bytes is refused, not read past.
        assert!(planar_to_packed(&codes, &scales[..4], row_elems, &mut back).is_err());
        assert!(packed_to_planar(&packed[..70], row_elems, &mut codes, &mut scales).is_err());
    }

    #[test]
    fn a_ring_splits_at_the_wrap() {
        let ring = Ring { rows: 512 };
        assert_eq!(
            ring.spans(400, 300),
            vec![span(400, 400, 112), span(512, 0, 188)]
        );
        assert_eq!(ring.oldest_resident(700), 188);
        assert_eq!(ring.oldest_resident(300), 0);
        assert_eq!(Linear.spans(5, 9), vec![span(5, 5, 9)]);
    }

    #[test]
    fn a_slot_map_splits_per_page_and_follows_the_table() {
        let be = infr_cpu::CpuBackend::new();
        let (_pool, _a, b) = two_slots(&be);
        assert_eq!(
            b.map.spans(500, 600),
            vec![
                span(500, 2 * PAGE_ROWS + 500, 12),
                span(512, 3 * PAGE_ROWS, 512),
                span(1024, 4 * PAGE_ROWS, 76),
            ]
        );
    }

    /// A block lands at its own ring position and wraps at the capacity, so a prefix longer than
    /// the ring still leaves the newest rows where attention reads them.
    #[test]
    fn a_ring_round_trip_wraps_at_its_capacity() {
        let be = infr_cpu::CpuBackend::new();
        let buf = be.alloc(4 * 512, BufferUsage::KvCache).expect("buffer");
        let ring = Ring { rows: 512 };
        let mut rows = Vec::with_capacity(300 * 4);
        for i in 0..300u32 {
            rows.extend_from_slice(&i.to_le_bytes());
        }
        write_rows(&be, buf.as_ref(), DType::F32, 1, &ring, 400..700, &rows).expect("write");
        let mut back = vec![0u8; rows.len()];
        read_rows(&be, buf.as_ref(), DType::F32, 1, &ring, 400..700, &mut back).expect("read");
        assert_eq!(back, rows);
        let mut out = vec![0u8; 4 * 512];
        be.download(buf.as_ref(), &mut out).expect("download");
        let at = |row: usize| u32::from_le_bytes(out[row * 4..row * 4 + 4].try_into().unwrap());
        assert_eq!(at(400), 0);
        assert_eq!(at(511), 111);
        assert_eq!(at(0), 112);
        assert_eq!(at(187), 299);
        assert_eq!(at(188), 0, "nothing past the block");
    }

    #[test]
    fn a_linear_copy_lands_on_the_mapped_pages() {
        let be = infr_cpu::CpuBackend::new();
        let (pool, _a, b) = two_slots(&be);
        let row = 32;
        let row_bytes = kv_fmt_bytes(DType::F16, row);
        let n = 600;
        let mut data = Vec::with_capacity(n * row_bytes);
        for r in 0..n as u16 {
            for _ in 0..row {
                data.extend_from_slice(&(r + 1).to_le_bytes());
            }
        }
        let src = be.alloc(data.len(), BufferUsage::KvCache).unwrap();
        be.upload(src.as_ref(), &data).unwrap();
        let k = pool.k(0).unwrap().clone();
        copy_rows(
            &be,
            src.as_ref(),
            &Linear,
            k.as_ref(),
            &b.map,
            DType::F16,
            row,
            0..n,
        )
        .unwrap();
        let mut whole = vec![0u8; pool.total_rows() * row_bytes];
        be.download(k.as_ref(), &mut whole).unwrap();
        for r in [0, 1, 511, 512, 599] {
            let phys = b.map.page(r / PAGE_ROWS) as usize * PAGE_ROWS + r % PAGE_ROWS;
            assert_eq!(
                &whole[phys * row_bytes..(phys + 1) * row_bytes],
                &data[r * row_bytes..(r + 1) * row_bytes],
                "row {r}"
            );
        }
        let page = PAGE_ROWS * row_bytes;
        assert!(
            whole[page..2 * page].iter().all(|&x| x == 0),
            "the other slot's page is untouched"
        );
    }
}
