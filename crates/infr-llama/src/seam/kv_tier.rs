//! KV extents behind one trait, whichever memory holds them.
//!
//! A [`KvStore`] holds extents: runs of rows for every layer, addressed by a `u64` id. A device
//! slot ([`DeviceKv`]) and a host map ([`RamKv`]) both implement it, so moving a prefix between
//! them is a [`get`](KvStore::get) into [`KvRows`] and a [`put`](KvStore::put) out of it. Attention
//! never goes through the trait; it binds the slot's buffers directly.

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use infr_core::backend::{Backend, Buffer};
use infr_core::DType;
use redb::{Database, Durability, ReadableDatabase, ReadableTable, TableDefinition};

use super::{
    kv_row_elems,
    rows::{read_rows, row_bytes, write_rows, RowLayout},
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
    #[cfg(test)]
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

/// FNV-1a over the bytes a codec writes, so damaged or truncated data is discarded.
pub(crate) struct Fnv(pub(crate) u64);

impl Fnv {
    pub(crate) fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    pub(crate) fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// Version of [`encode_rows`], [`encode_meta`] and the format blob.
const CODEC_VERSION: u8 = 1;

struct Writer {
    buf: Vec<u8>,
    h: Fnv,
}

impl Writer {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            h: Fnv::new(),
        }
    }

    fn u8(&mut self, v: u8) {
        self.buf.push(v);
        self.h.bytes(&[v]);
    }

    fn u32(&mut self, v: u32) {
        self.raw(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.raw(&v.to_le_bytes());
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        self.h.bytes(bytes);
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.u32(u32::try_from(bytes.len()).expect("a tier blob fits in u32"));
        self.raw(bytes);
    }

    fn finish(mut self) -> Vec<u8> {
        let h = self.h.0.to_le_bytes();
        self.buf.extend_from_slice(&h);
        self.buf
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
    h: Fnv,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            at: 0,
            h: Fnv::new(),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.at + n <= self.buf.len(), "kv tier blob is truncated");
        let bytes = &self.buf[self.at..self.at + n];
        self.h.bytes(bytes);
        self.at += n;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn bytes(&mut self) -> Result<Vec<u8>> {
        let n = self.u32()? as usize;
        Ok(self.take(n)?.to_vec())
    }

    fn finish(self) -> Result<()> {
        ensure!(
            self.at + 8 == self.buf.len(),
            "kv tier blob has trailing bytes"
        );
        let stored = u64::from_le_bytes(self.buf[self.at..].try_into()?);
        ensure!(stored == self.h.0, "kv tier blob failed its checksum");
        Ok(())
    }
}

/// Encode extent `id`'s rows, whose first row is `first`, for a byte-oriented store.
pub(crate) fn encode_rows(first: usize, rows: &KvRows) -> Vec<u8> {
    let mut w = Writer::new();
    w.u8(CODEC_VERSION);
    w.u64(first as u64);
    w.u64(rows.rows as u64);
    w.u32(rows.layers.len() as u32);
    for layer in &rows.layers {
        w.u32(layer.runs.len() as u32);
        for run in &layer.runs {
            w.u64(run.start as u64);
            w.u64(run.end as u64);
        }
        w.bytes(&layer.k);
        w.bytes(&layer.v);
    }
    match &rows.snap {
        None => w.u8(0),
        Some(snap) => {
            w.u8(1);
            w.u64(snap.pos as u64);
            w.u32(snap.k.len() as u32);
            for (l, bytes) in &snap.k {
                w.u32(*l as u32);
                w.bytes(bytes);
            }
            w.u32(snap.v.len() as u32);
            for (l, bytes) in &snap.v {
                w.u32(*l as u32);
                w.bytes(bytes);
            }
        }
    }
    w.finish()
}

/// Decode an extent and its first row. Every run survives, including a layer that holds only
/// some of the extent's rows.
pub(crate) fn decode_rows(buf: &[u8]) -> Result<(usize, KvRows)> {
    let mut r = Reader::new(buf);
    ensure!(r.u8()? == CODEC_VERSION, "kv tier blob version mismatch");
    let first = usize::try_from(r.u64()?)?;
    let rows = usize::try_from(r.u64()?)?;
    let n = r.u32()? as usize;
    ensure!(n <= 4096, "kv tier blob claims {n} layers");
    let mut layers = Vec::with_capacity(n);
    for _ in 0..n {
        let nr = r.u32()? as usize;
        ensure!(nr <= rows.max(1), "kv tier blob claims {nr} runs");
        let mut runs = Vec::with_capacity(nr);
        let mut held = 0usize;
        let mut prev = 0usize;
        for _ in 0..nr {
            let start = usize::try_from(r.u64()?)?;
            let end = usize::try_from(r.u64()?)?;
            ensure!(
                start >= prev && start < end && end <= rows,
                "kv tier blob has a bad run"
            );
            held += end - start;
            prev = end;
            runs.push(start..end);
        }
        let k = r.bytes()?;
        let v = r.bytes()?;
        if held == 0 {
            ensure!(k.is_empty() && v.is_empty(), "kv tier layer holds no rows");
        } else {
            ensure!(
                k.len().is_multiple_of(held) && v.len().is_multiple_of(held),
                "kv tier layer has the wrong byte count"
            );
        }
        layers.push(LayerRows { k, v, runs });
    }
    let snap = match r.u8()? {
        0 => None,
        1 => {
            let pos = usize::try_from(r.u64()?)?;
            let mut sides = Vec::new();
            let mut bytes = 0usize;
            for _ in 0..2 {
                let entries = r.u32()? as usize;
                ensure!(entries <= 4096, "kv tier snapshot claims {entries} layers");
                let mut side = Vec::with_capacity(entries);
                for _ in 0..entries {
                    let l = r.u32()? as usize;
                    let b = r.bytes()?;
                    bytes += b.len();
                    side.push((l, b));
                }
                sides.push(side);
            }
            let v = sides.pop().expect("two sides");
            let k = sides.pop().expect("two sides");
            Some(Arc::new(RecurrentSnap { pos, k, v, bytes }))
        }
        v => bail!("kv tier blob names snapshot state {v}"),
    };
    r.finish()?;
    Ok((first, KvRows { rows, layers, snap }))
}

/// Encode a block's index entry: its tokens, whether a snapshot rides the extent, and its bytes.
pub(crate) fn encode_meta(tokens: &[u32], snap: bool, bytes: usize) -> Vec<u8> {
    let mut w = Writer::new();
    w.u8(CODEC_VERSION);
    w.u32(tokens.len() as u32);
    for &t in tokens {
        w.u32(t);
    }
    w.u8(u8::from(snap));
    w.u64(bytes as u64);
    w.finish()
}

/// Decode a block's index entry.
pub(crate) fn decode_meta(buf: &[u8]) -> Result<(Vec<u32>, bool, usize)> {
    let mut r = Reader::new(buf);
    ensure!(r.u8()? == CODEC_VERSION, "kv tier meta version mismatch");
    let n = r.u32()? as usize;
    ensure!(n <= 1 << 20, "kv tier meta claims {n} tokens");
    let mut tokens = Vec::with_capacity(n);
    for _ in 0..n {
        tokens.push(r.u32()?);
    }
    let snap = match r.u8()? {
        0 => false,
        1 => true,
        v => bail!("kv tier meta names snapshot state {v}"),
    };
    let bytes = usize::try_from(r.u64()?)?;
    r.finish()?;
    Ok((tokens, snap, bytes))
}

/// Extents of KV rows addressed by id. A call moves every layer at once.
///
/// Ids [`FORMATS_META_ID`] and [`RESERVED_META_ID`] are reserved for side metadata and never name
/// an extent.
pub(crate) trait KvStore: Send {
    /// Append extent `id`'s rows `rows` to `out` (see [`KvRows::append`]). `Ok(false)` when the
    /// store does not hold the extent.
    fn get(&mut self, id: u64, rows: Range<usize>, out: &mut KvRows) -> Result<bool>;
    /// Store `rows` as extent `id`'s rows from `first`. A host store replaces the extent; a
    /// device store writes into its slot.
    fn put(&mut self, id: u64, first: usize, rows: KvRows) -> Result<()>;
    fn remove(&mut self, id: u64) -> Result<()>;
    fn contains(&self, id: u64) -> bool;

    /// Ids of the extents the store holds.
    fn extents(&mut self) -> Result<Vec<u64>> {
        Ok(Vec::new())
    }

    /// The side metadata stored with extent `id`, opaque to the store.
    fn meta(&mut self, id: u64) -> Result<Option<Vec<u8>>> {
        let _ = id;
        Ok(None)
    }

    /// Replace extent `id`'s side metadata.
    fn set_meta(&mut self, id: u64, meta: &[u8]) -> Result<()> {
        let _ = (id, meta);
        Ok(())
    }

    /// The side metadata of `ids`, in the same order. A store that can fetch a batch in one
    /// round trip overrides this.
    fn metas(&mut self, ids: &[u64]) -> Result<Vec<Option<Vec<u8>>>> {
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            out.push(self.meta(id)?);
        }
        Ok(out)
    }

    /// The extent's rows when they sit in host memory, so a reader can skip the copy.
    #[cfg_attr(not(test), allow(dead_code))]
    fn host(&self, id: u64) -> Option<&KvRows> {
        let _ = id;
        None
    }
}

/// The formats blob's meta entry. No extent may use this id.
pub(crate) const FORMATS_META_ID: u64 = u64::MAX;

/// A second side-metadata id, kept out of the extent namespace.
pub(crate) const RESERVED_META_ID: u64 = u64::MAX - 1;

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

    fn extents(&mut self) -> Result<Vec<u64>> {
        (**self).extents()
    }

    fn meta(&mut self, id: u64) -> Result<Option<Vec<u8>>> {
        (**self).meta(id)
    }

    fn set_meta(&mut self, id: u64, meta: &[u8]) -> Result<()> {
        (**self).set_meta(id, meta)
    }

    fn metas(&mut self, ids: &[u64]) -> Result<Vec<Option<Vec<u8>>>> {
        (**self).metas(ids)
    }

    fn host(&self, id: u64) -> Option<&KvRows> {
        (**self).host(id)
    }
}

/// An extent taken out of a budgeted store, with the side metadata that belonged to it.
pub(crate) struct Taken {
    pub(crate) id: u64,
    pub(crate) first: usize,
    pub(crate) rows: KvRows,
    pub(crate) meta: Option<Vec<u8>>,
}

/// A store that reports the bytes it holds and evicts its least-recently-used extent.
pub(crate) trait BudgetedStore: KvStore {
    /// Bytes of every extent held.
    fn used(&self) -> usize;
    /// Bytes the store keeps before an eviction; `usize::MAX` when unbounded.
    fn capacity(&self) -> usize;
    /// Remove the least-recently-used extent and hand back its rows and metadata.
    fn take_lru(&mut self) -> Option<Taken>;
}

/// One layer of a device slot: its buffers, formats and where its rows sit.
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

    fn host(&self, id: u64) -> Option<&KvRows> {
        let _ = id;
        None
    }
}

impl DeviceKv<'_> {
    /// Layer `l`'s buffers and layout, or `None` for a layer whose state travels in the snapshot.
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

/// One extent in a [`RamKv`].
struct RamExtent {
    first: usize,
    rows: KvRows,
    used: u64,
}

/// Extents in host memory. A zero capacity is unbounded; eviction is the caller's, through
/// [`BudgetedStore::take_lru`].
pub(crate) struct RamKv {
    cap: usize,
    clock: u64,
    extents: HashMap<u64, RamExtent>,
    metas: HashMap<u64, Vec<u8>>,
}

impl Default for RamKv {
    fn default() -> Self {
        Self::new(0)
    }
}

impl RamKv {
    /// A store holding `cap_bytes`, or unbounded when it is zero.
    pub(crate) fn new(cap_bytes: usize) -> Self {
        Self {
            cap: cap_bytes,
            clock: 0,
            extents: HashMap::new(),
            metas: HashMap::new(),
        }
    }

    fn stamp(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }
}

impl KvStore for RamKv {
    fn get(&mut self, id: u64, rows: Range<usize>, out: &mut KvRows) -> Result<bool> {
        let Some(ext) = self.extents.get_mut(&id) else {
            return Ok(false);
        };
        ensure!(
            rows.start >= ext.first,
            "kv rows {rows:?} before extent {id:x} starts at {}",
            ext.first
        );
        ensure!(
            rows.end <= ext.first + ext.rows.rows,
            "kv rows {rows:?} past extent {id:x} of {} rows",
            ext.rows.rows
        );
        let (first, rowsref) = (ext.first, &ext.rows);
        let used = self.clock + 1;
        out.append(rowsref, rows.start - first..rows.end - first)?;
        self.clock = used;
        if let Some(e) = self.extents.get_mut(&id) {
            e.used = used;
        }
        Ok(true)
    }

    fn put(&mut self, id: u64, first: usize, rows: KvRows) -> Result<()> {
        let used = self.stamp();
        self.extents.insert(id, RamExtent { first, rows, used });
        Ok(())
    }

    fn remove(&mut self, id: u64) -> Result<()> {
        self.extents.remove(&id);
        self.metas.remove(&id);
        Ok(())
    }

    fn contains(&self, id: u64) -> bool {
        self.extents.contains_key(&id)
    }

    fn extents(&mut self) -> Result<Vec<u64>> {
        Ok(self.extents.keys().copied().collect())
    }

    fn meta(&mut self, id: u64) -> Result<Option<Vec<u8>>> {
        Ok(self.metas.get(&id).cloned())
    }

    fn set_meta(&mut self, id: u64, meta: &[u8]) -> Result<()> {
        self.metas.insert(id, meta.to_vec());
        Ok(())
    }

    fn host(&self, id: u64) -> Option<&KvRows> {
        self.extents.get(&id).map(|e| &e.rows)
    }
}

impl BudgetedStore for RamKv {
    fn used(&self) -> usize {
        self.extents.values().map(|e| e.rows.bytes()).sum()
    }

    fn capacity(&self) -> usize {
        if self.cap == 0 {
            usize::MAX
        } else {
            self.cap
        }
    }

    fn take_lru(&mut self) -> Option<Taken> {
        let id = *self
            .extents
            .iter()
            .min_by_key(|(_, e)| e.used)
            .map(|(id, _)| id)?;
        let ext = self.extents.remove(&id)?;
        let meta = self.metas.remove(&id);
        Some(Taken {
            id,
            first: ext.first,
            rows: ext.rows,
            meta,
        })
    }
}

/// A fast budgeted store in front of a slower one: reads fall through and promote, writes demote
/// whatever no longer fits, so the pair reads as one store.
pub(crate) struct Tiered<Fast: BudgetedStore, Slow: KvStore> {
    fast: Fast,
    slow: Slow,
}

impl<Fast: BudgetedStore, Slow: KvStore> Tiered<Fast, Slow> {
    pub(crate) fn new(fast: Fast, slow: Slow) -> Self {
        Self { fast, slow }
    }

    /// Demote the least recently used extents until the fast tier fits its budget again, metadata
    /// included, and keep one that fits nowhere: a block bigger than the whole budget stays put.
    fn settle(&mut self) -> Result<()> {
        while self.fast.used() > self.fast.capacity() {
            let Some(taken) = self.fast.take_lru() else {
                break;
            };
            if self.fast.used() == 0 {
                self.fast.put(taken.id, taken.first, taken.rows)?;
                if let Some(meta) = taken.meta {
                    self.fast.set_meta(taken.id, &meta)?;
                }
                break;
            }
            let id = taken.id;
            self.slow.put(id, taken.first, taken.rows)?;
            if let Some(meta) = taken.meta {
                self.slow.set_meta(id, &meta)?;
            }
        }
        Ok(())
    }
}

impl<Fast: BudgetedStore, Slow: KvStore> KvStore for Tiered<Fast, Slow> {
    fn get(&mut self, id: u64, rows: Range<usize>, out: &mut KvRows) -> Result<bool> {
        if self.fast.get(id, rows.clone(), out)? {
            return Ok(true);
        }
        let mut hit = KvRows::default();
        if !self.slow.get(id, rows.clone(), &mut hit)? {
            return Ok(false);
        }
        // A read from row 0 is a whole block extent, which is what the cache stores.
        if rows.start == 0 && !rows.is_empty() {
            self.fast.put(id, 0, hit)?;
            self.settle()?;
            return self.fast.get(id, rows, out);
        }
        out.append(&hit, 0..hit.rows)?;
        Ok(true)
    }

    fn put(&mut self, id: u64, first: usize, rows: KvRows) -> Result<()> {
        // The fast copy is the newest, so an older one below it must not come back on a restart.
        if self.slow.contains(id) {
            self.slow.remove(id)?;
        }
        self.fast.put(id, first, rows)?;
        self.settle()
    }

    fn remove(&mut self, id: u64) -> Result<()> {
        self.fast.remove(id)?;
        self.slow.remove(id)
    }

    fn contains(&self, id: u64) -> bool {
        self.fast.contains(id) || self.slow.contains(id)
    }

    fn extents(&mut self) -> Result<Vec<u64>> {
        let mut ids = self.fast.extents()?;
        for id in self.slow.extents()? {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        Ok(ids)
    }

    fn meta(&mut self, id: u64) -> Result<Option<Vec<u8>>> {
        match self.fast.meta(id)? {
            Some(meta) => Ok(Some(meta)),
            None => self.slow.meta(id),
        }
    }

    fn set_meta(&mut self, id: u64, meta: &[u8]) -> Result<()> {
        self.fast.set_meta(id, meta)?;
        // A reserved entry describes the whole cache, so every tier that persists keeps it.
        if id == FORMATS_META_ID || self.slow.contains(id) {
            self.slow.set_meta(id, meta)?;
        }
        Ok(())
    }

    fn metas(&mut self, ids: &[u64]) -> Result<Vec<Option<Vec<u8>>>> {
        let mut out = self.fast.metas(ids)?;
        let missing: Vec<u64> = ids
            .iter()
            .zip(&out)
            .filter(|(_, m)| m.is_none())
            .map(|(&id, _)| id)
            .collect();
        if missing.is_empty() {
            return Ok(out);
        }
        let mut slow = self.slow.metas(&missing)?.into_iter();
        for slot in out.iter_mut().filter(|m| m.is_none()) {
            *slot = slow.next().unwrap_or(None);
        }
        Ok(out)
    }

    fn host(&self, id: u64) -> Option<&KvRows> {
        self.fast.host(id)
    }
}

#[cfg(test)]
impl<Fast: BudgetedStore, Slow: KvStore> Tiered<Fast, Slow> {
    pub(crate) fn fast(&self) -> &Fast {
        &self.fast
    }

    pub(crate) fn slow(&self) -> &Slow {
        &self.slow
    }
}

/// A budgeted store with nothing below it: a write drops the least recently used extents until it
/// fits again, so a cache over host memory alone stays inside `kv.store_mb`.
pub(crate) struct Bounded<S: BudgetedStore> {
    inner: S,
}

impl<S: BudgetedStore> Bounded<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: BudgetedStore> KvStore for Bounded<S> {
    fn get(&mut self, id: u64, rows: Range<usize>, out: &mut KvRows) -> Result<bool> {
        self.inner.get(id, rows, out)
    }

    fn put(&mut self, id: u64, first: usize, rows: KvRows) -> Result<()> {
        self.inner.put(id, first, rows)?;
        while self.inner.used() > self.inner.capacity() && self.inner.take_lru().is_some() {}
        Ok(())
    }

    fn remove(&mut self, id: u64) -> Result<()> {
        self.inner.remove(id)
    }

    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn extents(&mut self) -> Result<Vec<u64>> {
        self.inner.extents()
    }

    fn meta(&mut self, id: u64) -> Result<Option<Vec<u8>>> {
        self.inner.meta(id)
    }

    fn set_meta(&mut self, id: u64, meta: &[u8]) -> Result<()> {
        self.inner.set_meta(id, meta)
    }

    fn metas(&mut self, ids: &[u64]) -> Result<Vec<Option<Vec<u8>>>> {
        self.inner.metas(ids)
    }

    fn host(&self, id: u64) -> Option<&KvRows> {
        self.inner.host(id)
    }
}

pub(crate) const DISK_EXTENTS: TableDefinition<'static, (u64, u64), &'static [u8]> =
    TableDefinition::new("kv_extents");
const DISK_METAS: TableDefinition<'static, (u64, u64), &'static [u8]> =
    TableDefinition::new("kv_metas");

/// One extent tracked in memory: its bytes and its last use.
struct DiskExtent {
    bytes: usize,
    used: u64,
}

/// Extents in one redb database, under one model key. A commit is not flushed (`Durability::None`)
/// because a lost extent is only a cache miss; `Drop` flushes once.
pub(crate) struct DiskKv {
    db: Database,
    key: u64,
    cap: usize,
    clock: u64,
    extents: HashMap<u64, DiskExtent>,
    #[cfg(test)]
    reads: usize,
}

impl DiskKv {
    /// Open (or create) `path` for the extents of `key`, keeping `cap_bytes`.
    pub(crate) fn open(path: &Path, key: u64, cap_bytes: usize) -> Result<Self> {
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let db = Database::create(path)?;
        {
            let txn = db.begin_write()?;
            txn.open_table(DISK_EXTENTS)?;
            txn.open_table(DISK_METAS)?;
            txn.commit()?;
        }
        let mut kv = Self {
            db,
            key,
            cap: cap_bytes,
            clock: 0,
            extents: HashMap::new(),
            #[cfg(test)]
            reads: 0,
        };
        kv.scan()?;
        Ok(kv)
    }

    /// Rebuild the in-memory LRU from the keys of this model's extents and their meta records.
    fn scan(&mut self) -> Result<()> {
        let mut ids = Vec::new();
        {
            let txn = self.db.begin_read()?;
            let table = txn.open_table(DISK_EXTENTS)?;
            for item in table.range((self.key, 0u64)..=(self.key, u64::MAX))? {
                let (k, _) = item?;
                let (_, id) = k.value();
                ids.push(id);
            }
        }
        let (orphan_extents, orphan_metas): (Vec<u64>, Vec<u64>) = {
            let txn = self.db.begin_read()?;
            let extents = txn.open_table(DISK_EXTENTS)?;
            let metas = txn.open_table(DISK_METAS)?;
            let mut orphan_extents = Vec::new();
            for id in &ids {
                match metas.get((self.key, *id))? {
                    Some(v) => {
                        let bytes = record_bytes(v.value());
                        self.clock += 1;
                        self.extents.insert(
                            *id,
                            DiskExtent {
                                bytes,
                                used: self.clock,
                            },
                        );
                    }
                    None => orphan_extents.push(*id),
                }
            }
            let mut orphan_metas = Vec::new();
            for item in metas.range((self.key, 0u64)..=(self.key, u64::MAX))? {
                let (k, _) = item?;
                let (_, id) = k.value();
                if id != FORMATS_META_ID
                    && id != RESERVED_META_ID
                    && extents.get((self.key, id))?.is_none()
                {
                    orphan_metas.push(id);
                }
            }
            (orphan_extents, orphan_metas)
        };
        if !orphan_extents.is_empty() || !orphan_metas.is_empty() {
            let mut txn = self.db.begin_write()?;
            txn.set_durability(Durability::None)?;
            {
                let mut table = txn.open_table(DISK_EXTENTS)?;
                for id in orphan_extents {
                    table.remove((self.key, id))?;
                }
            }
            {
                let mut table = txn.open_table(DISK_METAS)?;
                for id in orphan_metas {
                    table.remove((self.key, id))?;
                }
            }
            txn.commit()?;
        }
        Ok(())
    }

    fn stamp(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// The extent's rows, without stamping its use.
    fn load(&self, id: u64) -> Result<Option<(usize, KvRows)>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DISK_EXTENTS)?;
        let Some(v) = table.get((self.key, id))? else {
            return Ok(None);
        };
        Ok(Some(decode_rows(v.value())?))
    }

    fn write_meta(&self, id: u64, bytes: usize, meta: &[u8]) -> Result<()> {
        let mut record = Vec::with_capacity(8 + meta.len());
        record.extend_from_slice(&(bytes as u64).to_le_bytes());
        record.extend_from_slice(meta);
        let mut txn = self.db.begin_write()?;
        txn.set_durability(Durability::None)?;
        {
            let mut table = txn.open_table(DISK_METAS)?;
            table.insert((self.key, id), record.as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn block_reads(&self) -> usize {
        self.reads
    }
}

/// The byte count at the front of a disk meta record.
fn record_bytes(record: &[u8]) -> usize {
    record
        .get(..8)
        .and_then(|b| b.try_into().ok())
        .map(u64::from_le_bytes)
        .map_or(0, |b| b as usize)
}

/// The cache's metadata after the byte count at the front of a disk meta record.
fn record_meta(record: &[u8]) -> &[u8] {
    record.get(8..).unwrap_or(&[])
}

impl KvStore for DiskKv {
    fn get(&mut self, id: u64, rows: Range<usize>, out: &mut KvRows) -> Result<bool> {
        #[cfg(test)]
        {
            self.reads += 1;
        }
        let Some((first, ext)) = self.load(id)? else {
            return Ok(false);
        };
        ensure!(
            rows.start >= first,
            "kv rows {rows:?} before extent {id:x} starts at {first}"
        );
        ensure!(
            rows.end <= first + ext.rows,
            "kv rows {rows:?} past extent {id:x} of {} rows",
            ext.rows
        );
        out.append(&ext, rows.start - first..rows.end - first)?;
        let used = self.stamp();
        if let Some(e) = self.extents.get_mut(&id) {
            e.used = used;
        }
        Ok(true)
    }

    fn put(&mut self, id: u64, first: usize, rows: KvRows) -> Result<()> {
        let bytes = rows.bytes();
        let blob = encode_rows(first, &rows);
        let mut txn = self.db.begin_write()?;
        txn.set_durability(Durability::None)?;
        {
            let mut table = txn.open_table(DISK_EXTENTS)?;
            table.insert((self.key, id), blob.as_slice())?;
        }
        {
            let mut table = txn.open_table(DISK_METAS)?;
            let meta = table
                .get((self.key, id))?
                .map(|v| record_meta(v.value()).to_vec())
                .unwrap_or_default();
            let mut record = Vec::with_capacity(8 + meta.len());
            record.extend_from_slice(&(bytes as u64).to_le_bytes());
            record.extend_from_slice(&meta);
            table.insert((self.key, id), record.as_slice())?;
        }
        txn.commit()?;
        let used = self.stamp();
        self.extents.insert(id, DiskExtent { bytes, used });
        Ok(())
    }

    fn remove(&mut self, id: u64) -> Result<()> {
        let mut txn = self.db.begin_write()?;
        txn.set_durability(Durability::None)?;
        {
            let mut table = txn.open_table(DISK_EXTENTS)?;
            table.remove((self.key, id))?;
        }
        {
            let mut table = txn.open_table(DISK_METAS)?;
            table.remove((self.key, id))?;
        }
        txn.commit()?;
        self.extents.remove(&id);
        Ok(())
    }

    fn contains(&self, id: u64) -> bool {
        self.extents.contains_key(&id)
    }

    fn extents(&mut self) -> Result<Vec<u64>> {
        Ok(self.extents.keys().copied().collect())
    }

    fn meta(&mut self, id: u64) -> Result<Option<Vec<u8>>> {
        let meta = {
            let txn = self.db.begin_read()?;
            let table = txn.open_table(DISK_METAS)?;
            let meta = table
                .get((self.key, id))?
                .map(|v| record_meta(v.value()).to_vec());
            meta
        };
        Ok(meta)
    }

    fn set_meta(&mut self, id: u64, meta: &[u8]) -> Result<()> {
        let bytes = self.extents.get(&id).map_or(0, |e| e.bytes);
        self.write_meta(id, bytes, meta)
    }

    fn metas(&mut self, ids: &[u64]) -> Result<Vec<Option<Vec<u8>>>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DISK_METAS)?;
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            let meta = table
                .get((self.key, id))?
                .map(|v| record_meta(v.value()).to_vec());
            out.push(meta);
        }
        Ok(out)
    }
}

impl BudgetedStore for DiskKv {
    fn used(&self) -> usize {
        self.extents.values().map(|e| e.bytes).sum()
    }

    fn capacity(&self) -> usize {
        if self.cap == 0 {
            usize::MAX
        } else {
            self.cap
        }
    }

    fn take_lru(&mut self) -> Option<Taken> {
        let id = *self
            .extents
            .iter()
            .min_by_key(|(_, e)| e.used)
            .map(|(id, _)| id)?;
        let loaded = self.load(id);
        let (first, rows) = match loaded {
            Ok(Some(v)) => v,
            Ok(None) => {
                self.extents.remove(&id);
                return None;
            }
            Err(e) => {
                tracing::warn!(error = %e, "kv store: dropping an unreadable disk extent");
                if let Err(e) = self.remove(id) {
                    tracing::warn!(error = %e, "kv store: removing a bad disk extent failed");
                }
                return None;
            }
        };
        let meta = self.meta(id).ok().flatten();
        if let Err(e) = self.remove(id) {
            tracing::warn!(error = %e, "kv store: demoting a disk extent failed");
            return None;
        }
        Some(Taken {
            id,
            first,
            rows,
            meta,
        })
    }
}

impl Drop for DiskKv {
    fn drop(&mut self) {
        match self.db.begin_write() {
            Ok(mut txn) => {
                if let Err(e) = txn.set_durability(Durability::Immediate) {
                    tracing::warn!(error = %e, "kv store: flushing the disk tier failed");
                    return;
                }
                if let Err(e) = txn.commit() {
                    tracing::warn!(error = %e, "kv store: flushing the disk tier failed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "kv store: flushing the disk tier failed"),
        }
    }
}

/// Extents in Redis or Valkey: one key per extent, one meta key per extent and one index set per
/// model. The server's `maxmemory` policy is the budget.
pub(crate) struct RedisKv {
    conn: redis::Connection,
    endpoint: String,
    key: u64,
    ids: std::collections::HashSet<u64>,
}

impl RedisKv {
    pub(crate) fn connect(url: &str, key: u64) -> Result<Self> {
        let client = redis::Client::open(url)?;
        let conn = client.get_connection()?;
        conn.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
        conn.set_write_timeout(Some(std::time::Duration::from_secs(30)))?;
        Ok(Self {
            conn,
            endpoint: crate::seam::kv_store::redact(url),
            key,
            ids: std::collections::HashSet::new(),
        })
    }

    pub(crate) fn describe(&self) -> String {
        self.endpoint.clone()
    }

    fn prefix(&self) -> String {
        format!("infr:kv:{:016x}", self.key)
    }

    fn key_extent(&self, id: u64) -> String {
        format!("{}:x{id:016x}", self.prefix())
    }

    fn key_meta(&self, id: u64) -> String {
        format!("{}:m{id:016x}", self.prefix())
    }

    fn key_index(&self) -> String {
        format!("{}:idx", self.prefix())
    }
}

impl KvStore for RedisKv {
    fn get(&mut self, id: u64, rows: Range<usize>, out: &mut KvRows) -> Result<bool> {
        let blob: Option<Vec<u8>> = redis::cmd("GET")
            .arg(self.key_extent(id))
            .query(&mut self.conn)?;
        let Some(blob) = blob else {
            return Ok(false);
        };
        let (first, ext) = decode_rows(&blob)?;
        ensure!(
            rows.start >= first,
            "kv rows {rows:?} before extent {id:x} starts at {first}"
        );
        ensure!(
            rows.end <= first + ext.rows,
            "kv rows {rows:?} past extent {id:x} of {} rows",
            ext.rows
        );
        out.append(&ext, rows.start - first..rows.end - first)?;
        Ok(true)
    }

    fn put(&mut self, id: u64, first: usize, rows: KvRows) -> Result<()> {
        let blob = encode_rows(first, &rows);
        let mut pipe = redis::pipe();
        pipe.cmd("SET").arg(self.key_extent(id)).arg(blob).ignore();
        pipe.cmd("SADD")
            .arg(self.key_index())
            .arg(format!("{id:016x}"))
            .ignore();
        let _: () = pipe.query(&mut self.conn)?;
        self.ids.insert(id);
        Ok(())
    }

    fn remove(&mut self, id: u64) -> Result<()> {
        let mut pipe = redis::pipe();
        pipe.cmd("DEL")
            .arg(self.key_extent(id))
            .arg(self.key_meta(id))
            .ignore();
        pipe.cmd("SREM")
            .arg(self.key_index())
            .arg(format!("{id:016x}"))
            .ignore();
        let _: () = pipe.query(&mut self.conn)?;
        self.ids.remove(&id);
        Ok(())
    }

    fn contains(&self, id: u64) -> bool {
        self.ids.contains(&id)
    }

    fn extents(&mut self) -> Result<Vec<u64>> {
        let ids: Vec<String> = redis::cmd("SMEMBERS")
            .arg(self.key_index())
            .query(&mut self.conn)?;
        let mut out: Vec<u64> = ids
            .iter()
            .filter_map(|id| u64::from_str_radix(id, 16).ok())
            .collect();
        out.sort_unstable();
        self.ids = out.iter().copied().collect();
        Ok(out)
    }

    fn meta(&mut self, id: u64) -> Result<Option<Vec<u8>>> {
        Ok(redis::cmd("GET")
            .arg(self.key_meta(id))
            .query(&mut self.conn)?)
    }

    fn set_meta(&mut self, id: u64, meta: &[u8]) -> Result<()> {
        let _: () = redis::cmd("SET")
            .arg(self.key_meta(id))
            .arg(meta)
            .query(&mut self.conn)?;
        Ok(())
    }

    fn metas(&mut self, ids: &[u64]) -> Result<Vec<Option<Vec<u8>>>> {
        let mut out = vec![None; ids.len()];
        for (chunk, slots) in ids.chunks(64).zip(out.chunks_mut(64)) {
            let keys: Vec<String> = chunk.iter().map(|&id| self.key_meta(id)).collect();
            let vals: Vec<Option<Vec<u8>>> = redis::cmd("MGET").arg(&keys).query(&mut self.conn)?;
            slots.clone_from_slice(&vals);
        }
        Ok(out)
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

    /// A whole extent moves into a slot and back out unchanged, which is what a restore and a
    /// capture do to it.
    #[test]
    fn a_device_slot_round_trips_an_extent() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = cfg();
        let src = slot(&be, &cfg, 100);
        let dst = SeamKv::for_test(&be, &cfg, 512, false, 64, DType::F16).unwrap();
        let (mut a, mut b) = (
            DeviceKv::new(1, &be, &cfg, &src),
            DeviceKv::new(2, &be, &cfg, &dst),
        );
        let mut want = KvRows::default();
        assert!(a.get(1, 10..60, &mut want).unwrap());
        b.put(2, 10, want.clone()).unwrap();
        let mut got = KvRows::default();
        assert!(b.get(2, 10..60, &mut got).unwrap());
        assert_eq!(got.layers, want.layers);
        let mut head = KvRows::default();
        b.get(2, 0..10, &mut head).unwrap();
        assert!(
            head.layers[0].k.iter().all(|&x| x == 0),
            "rows before the write"
        );
    }
}
