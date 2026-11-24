//! KV extents behind one trait, whichever memory holds them.
//!
//! A [`KvStore`] holds extents: runs of rows for every layer, addressed by a `u64` id. A device
//! slot ([`DeviceKv`]) and a host map ([`RamKv`]) both implement it, so moving a prefix between
//! them is a [`get`](KvStore::get) into [`KvRows`] and a [`put`](KvStore::put) out of it. Attention
//! never goes through the trait; it binds the slot's buffers directly.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use infr_core::backend::{Backend, Buffer};
use infr_core::DType;

use super::{
    kv_row_elems,
    rows::{copy_rows, read_rows, row_bytes, write_rows, RowLayout},
    weights::{RecurrentSnap, SeamKv},
};
use crate::Config;

/// One layer's rows inside a [`KvRows`].
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub(crate) struct LayerRows {
    /// The K rows of every run, back to back.
    pub(crate) k: Vec<u8>,
    /// The V rows of every run; empty for a layer with no V side.
    pub(crate) v: Vec<u8>,
    /// The rows `k`/`v` hold, relative to the first row of the [`KvRows`], ascending and disjoint.
    /// Rows outside every run are absent for this layer.
    pub(crate) runs: Vec<Range<usize>>,
}

impl LayerRows {
    fn held(&self) -> usize {
        self.runs.iter().map(|r| r.len()).sum()
    }

    fn push_run(&mut self, r: Range<usize>) {
        match self.runs.last_mut() {
            Some(last) if last.end == r.start => last.end = r.end,
            _ => self.runs.push(r),
        }
    }

    /// Append `src`'s rows inside `range`, placed at `base + (row - range.start)`.
    fn append(&mut self, src: &LayerRows, range: Range<usize>, base: usize) {
        let held = src.held();
        if held == 0 {
            return;
        }
        let (kb, vb) = (src.k.len() / held, src.v.len() / held);
        let mut at = 0;
        for run in &src.runs {
            let (lo, hi) = (run.start.max(range.start), run.end.min(range.end));
            if lo < hi {
                let off = at + (lo - run.start);
                let n = hi - lo;
                self.k.extend_from_slice(&src.k[off * kb..(off + n) * kb]);
                self.v.extend_from_slice(&src.v[off * vb..(off + n) * vb]);
                self.push_run(base + lo - range.start..base + hi - range.start);
            }
            at += run.len();
        }
    }

    /// Whether every row of `range` is held by some run.
    pub(crate) fn covers(&self, range: Range<usize>) -> bool {
        let mut at = range.start;
        for run in &self.runs {
            if run.start > at {
                return at >= range.end;
            }
            at = at.max(run.end);
        }
        at >= range.end
    }
}

/// Rows of every layer for one run of positions, plus the recurrent state at its end.
#[derive(Clone, Default, Debug)]
pub(crate) struct KvRows {
    /// Positions covered; a layer may hold fewer (see [`LayerRows::runs`]).
    pub(crate) rows: usize,
    /// Per layer in `SeamKv::kbufs` order, or empty when no layer holds rows.
    pub(crate) layers: Vec<LayerRows>,
    /// Every recurrent layer's state after the last row, when one was kept.
    pub(crate) snap: Option<Arc<RecurrentSnap>>,
}

impl KvRows {
    /// `rows` positions whose layers each hold either all of them or none (an empty `k`).
    pub(crate) fn whole(rows: usize, k: Vec<Vec<u8>>, v: Vec<Vec<u8>>) -> Self {
        let layers = k
            .into_iter()
            .zip(v)
            .map(|(k, v)| LayerRows {
                runs: if k.is_empty() || rows == 0 {
                    Vec::new()
                } else {
                    std::iter::once(0..rows).collect()
                },
                k,
                v,
            })
            .collect();
        Self {
            rows,
            layers,
            snap: None,
        }
    }

    /// Host bytes held, the snapshot included.
    pub(crate) fn bytes(&self) -> usize {
        let rows: usize = self.layers.iter().map(|l| l.k.len() + l.v.len()).sum();
        rows + self.snap.as_ref().map_or(0, |s| s.bytes)
    }

    /// Layer `l`'s bytes when it holds every row, `None` when it holds only some.
    pub(crate) fn whole_layer(&self, l: usize) -> Option<(&[u8], &[u8])> {
        let layer = self.layers.get(l)?;
        let whole = layer.runs.is_empty() || layer.held() == self.rows;
        whole.then_some((&layer.k, &layer.v))
    }

    /// Append `src`'s rows `range` after this one's last row. The snapshot becomes `src`'s when
    /// `range` ends at `src`'s end, and none otherwise.
    pub(crate) fn append(&mut self, src: &KvRows, range: Range<usize>) -> Result<()> {
        ensure!(
            range.start <= range.end && range.end <= src.rows,
            "kv rows {range:?} outside an extent of {} rows",
            src.rows
        );
        if self.layers.is_empty() {
            self.layers
                .resize_with(src.layers.len(), LayerRows::default);
        }
        ensure!(
            src.layers.is_empty() || src.layers.len() == self.layers.len(),
            "kv rows of {} layers appended to {}",
            src.layers.len(),
            self.layers.len()
        );
        let base = self.rows;
        for (dst, s) in self.layers.iter_mut().zip(&src.layers) {
            dst.append(s, range.clone(), base);
        }
        self.rows += range.len();
        self.snap = if range.end == src.rows {
            src.snap.clone()
        } else {
            None
        };
        Ok(())
    }
}

/// Extents of KV rows addressed by id. A call moves every layer at once.
pub(crate) trait KvStore {
    /// Append extent `id`'s rows `rows` to `out` (see [`KvRows::append`]). `Ok(false)` when the
    /// store does not hold the extent.
    fn get(&mut self, id: u64, rows: Range<usize>, out: &mut KvRows) -> Result<bool>;
    /// Store `rows` as extent `id`'s rows from `first`. A host store replaces the extent; a
    /// device store writes into its slot.
    fn put(&mut self, id: u64, first: usize, rows: KvRows) -> Result<()>;
    fn remove(&mut self, id: u64) -> Result<()>;
    fn contains(&self, id: u64) -> bool;

    /// The extent's rows when they sit in host memory, so a reader can skip the copy.
    fn host(&self, id: u64) -> Option<&KvRows> {
        let _ = id;
        None
    }

    /// This store as a device store, which lets [`copy`] stay on the device.
    fn as_device(&self) -> Option<&dyn DeviceStore> {
        None
    }
}

impl<S: KvStore + ?Sized> KvStore for Box<S> {
    fn get(&mut self, id: u64, rows: Range<usize>, out: &mut KvRows) -> Result<bool> {
        (**self).get(id, rows, out)
    }

    fn put(&mut self, id: u64, first: usize, rows: KvRows) -> Result<()> {
        (**self).put(id, first, rows)
    }

    fn remove(&mut self, id: u64) -> Result<()> {
        (**self).remove(id)
    }

    fn contains(&self, id: u64) -> bool {
        (**self).contains(id)
    }

    fn host(&self, id: u64) -> Option<&KvRows> {
        (**self).host(id)
    }

    fn as_device(&self) -> Option<&dyn DeviceStore> {
        (**self).as_device()
    }
}

/// One layer of a device store: its buffers, formats and where its rows sit.
pub(crate) struct DeviceLayer<'a> {
    pub(crate) k: &'a dyn Buffer,
    pub(crate) v: &'a dyn Buffer,
    pub(crate) k_fmt: DType,
    pub(crate) v_fmt: DType,
    pub(crate) k_row: usize,
    /// Zero for a layer with no V side.
    pub(crate) v_row: usize,
    pub(crate) layout: Box<dyn RowLayout + 'a>,
    /// The oldest row still resident; older rows were recycled by a ring.
    pub(crate) oldest: usize,
}

/// A store whose rows live in device buffers.
pub(crate) trait DeviceStore: KvStore {
    fn backend(&self) -> &dyn Backend;
    fn n_layer(&self) -> usize;
    /// Layer `l`, or `None` for a layer whose state travels in the snapshot.
    fn layer(&self, l: usize) -> Option<DeviceLayer<'_>>;
}

/// Copy extent `src_id`'s rows `rows` into extent `dst_id` at the same rows. Two device stores on
/// one backend copy on the device; anything else goes through host memory. `Ok(false)` when `src`
/// lacks the extent. A paged destination must own the pages it receives.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn copy(
    src: &mut dyn KvStore,
    src_id: u64,
    dst: &mut dyn KvStore,
    dst_id: u64,
    rows: Range<usize>,
) -> Result<bool> {
    if !src.contains(src_id) {
        return Ok(false);
    }
    if let (Some(s), Some(d)) = (src.as_device(), dst.as_device()) {
        if std::ptr::addr_eq(s.backend(), d.backend()) {
            copy_on_device(s, d, rows.clone())?;
            let mut state = KvRows::default();
            src.get(src_id, rows.end..rows.end, &mut state)?;
            if state.snap.is_some() {
                dst.put(dst_id, rows.end, state)?;
            }
            return Ok(true);
        }
    }
    let mut buf = KvRows::default();
    if !src.get(src_id, rows.clone(), &mut buf)? {
        return Ok(false);
    }
    dst.put(dst_id, rows.start, buf)?;
    Ok(true)
}

#[cfg_attr(not(test), allow(dead_code))]
fn copy_on_device(s: &dyn DeviceStore, d: &dyn DeviceStore, rows: Range<usize>) -> Result<()> {
    ensure!(
        s.n_layer() == d.n_layer(),
        "kv copy between {} and {} layers",
        s.n_layer(),
        d.n_layer()
    );
    let be = s.backend();
    for l in 0..s.n_layer() {
        let (a, b) = match (s.layer(l), d.layer(l)) {
            (Some(a), Some(b)) => (a, b),
            (None, None) => continue,
            _ => bail!("kv copy: layer {l} keeps rows on one side only"),
        };
        ensure!(
            (a.k_fmt, a.v_fmt, a.k_row, a.v_row) == (b.k_fmt, b.v_fmt, b.k_row, b.v_row),
            "kv copy: layer {l} formats differ"
        );
        let from = rows.start.max(a.oldest).min(rows.end);
        let span = from..rows.end;
        copy_rows(
            be,
            a.k,
            &*a.layout,
            b.k,
            &*b.layout,
            a.k_fmt,
            a.k_row,
            span.clone(),
        )?;
        if a.v_row > 0 {
            copy_rows(be, a.v, &*a.layout, b.v, &*b.layout, a.v_fmt, a.v_row, span)?;
        }
    }
    Ok(())
}

/// One slot's KV buffers as a store holding the single extent `id`, whose rows are the slot's
/// logical rows. A read skips rows a ring already recycled; a write into a paged slot needs its
/// pages private first. The slot's `cached` tokens are its owner's to update.
pub(crate) struct DeviceKv<'a> {
    id: u64,
    be: &'a dyn Backend,
    cfg: &'a Config,
    kv: &'a SeamKv,
}

impl<'a> DeviceKv<'a> {
    pub(crate) fn new(id: u64, be: &'a dyn Backend, cfg: &'a Config, kv: &'a SeamKv) -> Self {
        Self { id, be, cfg, kv }
    }

    fn check(&self, id: u64, end: usize) -> Result<()> {
        ensure!(id == self.id, "kv slot {} has no extent {id}", self.id);
        ensure!(
            end <= self.kv.max_ctx,
            "kv rows up to {end} past a slot of {}",
            self.kv.max_ctx
        );
        Ok(())
    }
}

impl KvStore for DeviceKv<'_> {
    fn get(&mut self, id: u64, rows: Range<usize>, out: &mut KvRows) -> Result<bool> {
        self.check(id, rows.end)?;
        let n = self.cfg.n_layer;
        if out.layers.is_empty() {
            out.layers.resize_with(n, LayerRows::default);
        }
        ensure!(
            out.layers.len() == n,
            "kv rows of {} layers read from {n}",
            out.layers.len()
        );
        let base = out.rows;
        for (l, dst) in out.layers.iter_mut().enumerate() {
            let Some(d) = self.layer(l) else { continue };
            let from = rows.start.max(d.oldest);
            if from >= rows.end {
                continue;
            }
            let span = from..rows.end;
            let at = dst.k.len();
            dst.k
                .resize(at + span.len() * row_bytes(d.k_fmt, d.k_row)?, 0);
            let (layout, be) = (&*d.layout, self.be);
            read_rows(
                be,
                d.k,
                d.k_fmt,
                d.k_row,
                layout,
                span.clone(),
                &mut dst.k[at..],
            )?;
            if d.v_row > 0 {
                let at = dst.v.len();
                dst.v
                    .resize(at + span.len() * row_bytes(d.v_fmt, d.v_row)?, 0);
                read_rows(
                    be,
                    d.v,
                    d.v_fmt,
                    d.v_row,
                    layout,
                    span.clone(),
                    &mut dst.v[at..],
                )?;
            }
            dst.push_run(base + from - rows.start..base + rows.len());
        }
        out.rows += rows.len();
        out.snap = self
            .kv
            .recurrent_snaps
            .iter()
            .find(|s| s.pos > 0 && s.pos == rows.end)
            .map(|s| Arc::new(s.clone()));
        Ok(true)
    }

    fn put(&mut self, id: u64, first: usize, rows: KvRows) -> Result<()> {
        self.check(id, first + rows.rows)?;
        ensure!(
            rows.layers.is_empty() || rows.layers.len() == self.cfg.n_layer,
            "kv rows of {} layers written to {}",
            rows.layers.len(),
            self.cfg.n_layer
        );
        for (l, src) in rows.layers.iter().enumerate() {
            let Some(d) = self.layer(l) else { continue };
            let (kb, vb) = (row_bytes(d.k_fmt, d.k_row)?, row_bytes(d.v_fmt, d.v_row)?);
            let held = src.held();
            ensure!(
                src.k.len() == held * kb && (d.v_row == 0 || src.v.len() == held * vb),
                "kv rows of layer {l} hold the wrong byte count"
            );
            let mut at = 0;
            for run in &src.runs {
                let span = first + run.start..first + run.end;
                let k = &src.k[at * kb..(at + run.len()) * kb];
                write_rows(self.be, d.k, d.k_fmt, d.k_row, &*d.layout, span.clone(), k)?;
                if d.v_row > 0 {
                    let v = &src.v[at * vb..(at + run.len()) * vb];
                    write_rows(self.be, d.v, d.v_fmt, d.v_row, &*d.layout, span, v)?;
                }
                at += run.len();
            }
        }
        if let Some(snap) = &rows.snap {
            for (l, bytes) in &snap.k {
                self.be.upload(self.kv.kbufs[*l].as_ref(), bytes)?;
            }
            for (l, bytes) in &snap.v {
                self.be.upload(self.kv.vbufs[*l].as_ref(), bytes)?;
            }
        }
        Ok(())
    }

    fn remove(&mut self, id: u64) -> Result<()> {
        bail!("kv slot {id}: a slot's rows are released by its owner")
    }

    fn contains(&self, id: u64) -> bool {
        id == self.id
    }

    fn as_device(&self) -> Option<&dyn DeviceStore> {
        Some(self)
    }
}

impl DeviceStore for DeviceKv<'_> {
    fn backend(&self) -> &dyn Backend {
        self.be
    }

    fn n_layer(&self) -> usize {
        self.cfg.n_layer
    }

    fn layer(&self, l: usize) -> Option<DeviceLayer<'_>> {
        if self.cfg.is_recurrent_layer(l) {
            return None;
        }
        let (k_row, v_row) = kv_row_elems(self.cfg, l);
        let layout = self.kv.row_layout(self.be, self.cfg, l);
        // Rows a chain wrote past `cached` still recycled the ring under them.
        // Rows a chain wrote past `cached` still recycled the ring under them.
        let oldest = layout.oldest_resident(self.kv.written_end.max(self.kv.cached.len()));
        Some(DeviceLayer {
            k: self.kv.kbufs[l].as_ref(),
            v: self.kv.vbufs[l].as_ref(),
            k_fmt: self.kv.k_fmt,
            v_fmt: self.kv.v_fmt,
            k_row,
            v_row,
            layout,
            oldest,
        })
    }
}

/// Extents in host memory.
#[derive(Default)]
pub(crate) struct RamKv {
    extents: HashMap<u64, (usize, KvRows)>,
}

impl KvStore for RamKv {
    fn get(&mut self, id: u64, rows: Range<usize>, out: &mut KvRows) -> Result<bool> {
        let Some((first, ext)) = self.extents.get(&id) else {
            return Ok(false);
        };
        ensure!(
            rows.start >= *first,
            "kv rows {rows:?} before extent {id:x} starts at {first}"
        );
        out.append(ext, rows.start - first..rows.end - first)?;
        Ok(true)
    }

    fn put(&mut self, id: u64, first: usize, rows: KvRows) -> Result<()> {
        self.extents.insert(id, (first, rows));
        Ok(())
    }

    fn remove(&mut self, id: u64) -> Result<()> {
        self.extents.remove(&id);
        Ok(())
    }

    fn contains(&self, id: u64) -> bool {
        self.extents.contains_key(&id)
    }

    fn host(&self, id: u64) -> Option<&KvRows> {
        self.extents.get(&id).map(|(_, rows)| rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, seed: u8, layers: usize) -> KvRows {
        let k = (0..layers)
            .map(|l| vec![seed.wrapping_add(l as u8); n * 4])
            .collect();
        let v = (0..layers).map(|_| vec![seed; n * 2]).collect();
        KvRows::whole(n, k, v)
    }

    fn one(r: Range<usize>) -> Vec<Range<usize>> {
        std::iter::once(r).collect()
    }

    #[test]
    fn ram_kv_round_trips_and_removes() {
        let mut s = RamKv::default();
        let a = rows(3, 1, 2);
        s.put(7, 0, a.clone()).unwrap();
        assert!(s.contains(7));
        let mut out = KvRows::default();
        assert!(s.get(7, 0..3, &mut out).unwrap());
        assert_eq!(out.layers, a.layers);
        assert_eq!(out.rows, 3);

        // A second extent appends after the first, merging the runs.
        s.put(8, 0, rows(2, 9, 2)).unwrap();
        assert!(s.get(8, 0..2, &mut out).unwrap());
        assert_eq!(out.rows, 5);
        assert_eq!(out.layers[0].runs, one(0..5));
        assert_eq!(&out.layers[1].k[12..], &[10u8; 8]);

        // A sub-range reads only its rows.
        let mut part = KvRows::default();
        assert!(s.get(7, 1..3, &mut part).unwrap());
        assert_eq!((part.rows, part.layers[0].k.len()), (2, 8));

        s.remove(7).unwrap();
        assert!(!s.contains(7));
        assert!(!s.get(7, 0..3, &mut out).unwrap());
    }

    #[test]
    fn an_absent_layer_breaks_the_run_and_coverage() {
        let mut out = KvRows::default();
        out.append(&rows(2, 1, 2), 0..2).unwrap();
        let mut gap = rows(2, 1, 2);
        gap.layers[1] = LayerRows::default();
        out.append(&gap, 0..2).unwrap();
        out.append(&rows(2, 1, 2), 0..2).unwrap();
        assert_eq!(out.layers[0].runs, one(0..6));
        assert_eq!(out.layers[1].runs, vec![0..2, 4..6]);
        assert_eq!(out.whole_layer(1), None);
        assert!(!out.layers[1].covers(0..6));
        assert!(out.layers[1].covers(4..6));

        let mut tail = KvRows::default();
        tail.append(&out, 1..4).unwrap();
        assert_eq!(tail.layers[1].runs, one(0..1));
        assert_eq!(tail.layers[0].runs, one(0..3));
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

    /// A linear f16 slot whose layer `l` row `r` holds `r + l` in every element.
    fn slot(be: &dyn Backend, cfg: &Config, n: usize) -> SeamKv {
        let mut kv = SeamKv::for_test(be, cfg, 512, false, 64, DType::F16).unwrap();
        for l in 0..cfg.n_layer {
            let data: Vec<u8> = (0..n)
                .flat_map(|r| ((r + l) as u16).to_le_bytes().repeat(32))
                .collect();
            for buf in [&kv.kbufs[l], &kv.vbufs[l]] {
                be.upload_at(buf.as_ref(), 0, &data).unwrap();
            }
        }
        kv.cached = vec![1; n];
        kv
    }

    #[test]
    fn copy_stays_on_the_device_or_goes_through_the_host() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = cfg();
        let src = slot(&be, &cfg, 100);
        let dst = SeamKv::for_test(&be, &cfg, 512, false, 64, DType::F16).unwrap();
        let (mut a, mut b) = (
            DeviceKv::new(1, &be, &cfg, &src),
            DeviceKv::new(2, &be, &cfg, &dst),
        );
        assert!(!copy(&mut a, 9, &mut b, 2, 0..10).unwrap());
        assert!(copy(&mut a, 1, &mut b, 2, 10..60).unwrap());
        let mut want = KvRows::default();
        a.get(1, 10..60, &mut want).unwrap();
        let mut got = KvRows::default();
        b.get(2, 10..60, &mut got).unwrap();
        assert_eq!(got.layers, want.layers);
        let mut head = KvRows::default();
        b.get(2, 0..10, &mut head).unwrap();
        assert!(
            head.layers[0].k.iter().all(|&x| x == 0),
            "rows before the copy"
        );

        let mut ram = RamKv::default();
        assert!(copy(&mut a, 1, &mut ram, 5, 10..60).unwrap());
        let mut back = KvRows::default();
        assert!(ram.get(5, 10..60, &mut back).unwrap());
        assert_eq!(back.layers, want.layers);
    }
}
