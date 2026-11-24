//! Where a layer's logical KV rows sit inside its buffer, and the row copies that follow them.
//!
//! A slot's cache keeps logical row `r` at one of three physical rows: `r` for a linear cache,
//! `r % rows` for a sliding-window ring, or inside the page the slot's table maps `r` to. Every
//! store, restore and page copy walks [`RowLayout::spans`] instead of re-deriving that mapping.
//! A row's byte offset is always `kv_fmt_bytes(fmt, physical_row * row_elems)`.

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
    for s in layout.spans(rows.start, len) {
        let at = (s.logical - rows.start) * rb;
        be.download_at(buf, s.physical * rb, &mut out[at..at + s.rows * rb])?;
    }
    Ok(())
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
    for s in layout.spans(rows.start, len) {
        let at = (s.logical - rows.start) * rb;
        be.upload_at(buf, s.physical * rb, &data[at..at + s.rows * rb])?;
    }
    Ok(())
}

/// Copy logical `rows` from `src` to `dst` on the device, one copy per run both layouts agree on.
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
            if !(same_buffer && from == to) {
                be.copy_buffer_at(src, from * rb, dst, to * rb, (hi - lo) * rb)?;
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
