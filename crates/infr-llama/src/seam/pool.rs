//! Session-wide KV page pool: the physical half of forward PagedAttention in the seam.
//!
//! One [`KvPool`] serves every conversation slot of a session out of a shared budget of physical
//! pages. A slot starts with no pages and grows as its sequence does; pages are handed back when
//! the slot is reset. Reads and stores resolve logical rows through the slot's page table, so a
//! request's memory is what it uses, not the model's whole window.
//!
//! A forked slot starts by SHARING its parent's prefix pages instead of copying them, and splits
//! the sharing one page at a time when it writes into shared rows. Physical page 0 is a zero page
//! the table uses for logical pages a slot has not allocated yet; attention never reads one.
//!
//! Page ids are token-granular and identical across layers: physical page `p` holds rows
//! `p*PAGE_ROWS..(p+1)*PAGE_ROWS` of EVERY pooled layer. That is what lets one table buffer per
//! SLOT (not per layer) drive every layer's reads.

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::budget::kv_fmt_bytes;
use infr_core::paging::KvPagesExhausted;
use infr_core::DType;

use crate::seam::rows::{copy_rows, Linear};
use crate::seam::{kv_row_elems, kv_side_bytes};
use crate::Config;
use infr_vulkan::paged::{PagePool, PAGE_ROWS};

/// Conversation slots one auto-sized pool covers. Two is where sharing pays (the acceptance case
/// is two conversations over one long shared prefix) and it stays at what two linear slots already
/// cost. A further fork finds no room and fails, which the slot allocators already handle by
/// recycling their LRU slot.
pub(crate) const POOL_BLOCKS: usize = 2;

/// Whether layer `l`'s cache can live in the page pool: reads and stores go through the page
/// table; SWA and recurrent layers are position-addressed private caches.
pub(crate) fn paged_layer(cfg: &Config, l: usize) -> bool {
    if cfg.is_recurrent_layer(l) {
        return false;
    }
    if cfg.deepseek2 || cfg.deepseek4 || cfg.is_lite || cfg.diffusion_gemma {
        return false;
    }
    if cfg.is_swa_layer(l) {
        return false;
    }
    let (k_row, v_row) = kv_row_elems(cfg, l);
    k_row > 0 && v_row > 0
}

/// Why a model has nothing to page, in the caller's words. `None` when at least one layer is
/// poolable.
pub(crate) fn no_poolable_layer(cfg: &Config) -> Option<&'static str> {
    if (0..cfg.n_layer).any(|l| paged_layer(cfg, l)) {
        None
    } else {
        Some(
            "no layer's KV cache is row-addressed (every layer is a sliding-window ring, a \
             recurrent mixer, or a compressed cache), so there is nothing to page",
        )
    }
}

/// The physical pages behind one session's conversation slots.
pub(crate) struct KvPool {
    /// Per layer: that layer's K pool buffer, `None` for a layer that keeps a private cache.
    k: Vec<Option<Arc<dyn Buffer>>>,
    v: Vec<Option<Arc<dyn Buffer>>>,
    /// Per-layer row width in elements, for the COW copies; 0 for a non-pooled layer.
    k_row: Vec<usize>,
    v_row: Vec<usize>,
    k_fmt: DType,
    v_fmt: DType,
    /// Reference ledger: `refs[p]` counts the slots pointing at physical page `p`. Page 0 is
    /// reserved forever as the table's unallocated filler.
    pages: Mutex<PagePool>,
    /// Logical pages per slot (the model window, rounded up to whole pages), and its row count.
    pps: usize,
    slot_rows: usize,
    /// Physical pages, page 0 included.
    total_pages: usize,
}

impl KvPool {
    /// Build a pool of `pages` physical pages (page 0 is the zero page). `max_ctx` sets the
    /// logical window each slot's table covers.
    pub(crate) fn new(
        be: &dyn Backend,
        cfg: &Config,
        max_ctx: usize,
        k_fmt: DType,
        v_fmt: DType,
        pages: usize,
    ) -> AResult<Arc<Self>> {
        if let Some(why) = no_poolable_layer(cfg) {
            return Err(anyhow!("{why}"));
        }
        // A TurboQuant cache has no page-aware store or dequant prepass, so a page table would
        // write rows the linear reads never find.
        let turbo = |f: DType| matches!(f, DType::Turbo2 | DType::Turbo3 | DType::Turbo4);
        if turbo(k_fmt) || turbo(v_fmt) {
            return Err(anyhow!(
                "a TurboQuant KV cache ({k_fmt:?} K / {v_fmt:?} V) has no page-aware store or \
                 dequant, so kv.paged stays off"
            ));
        }
        // A fork's tail copy and a page's copy-on-write move byte runs, which only exist when a
        // Q8_0 side packs rows.
        if (k_fmt == DType::Q8_0 || v_fmt == DType::Q8_0) && be.kv_q8_planar() {
            return Err(anyhow!(
                "a planar Q8_0 KV cache ({k_fmt:?} K / {v_fmt:?} V) has no packed page rows, so \
                 kv.paged stays off"
            ));
        }
        let pages = pages.max(2);
        let pps = max_ctx.div_ceil(PAGE_ROWS).max(1);
        let slot_rows = pps * PAGE_ROWS;
        let pool_rows = pages * PAGE_ROWS;
        let mut k = Vec::with_capacity(cfg.n_layer);
        let mut v = Vec::with_capacity(cfg.n_layer);
        let mut k_row = Vec::with_capacity(cfg.n_layer);
        let mut v_row = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let (kr, vr) = kv_row_elems(cfg, l);
            if !paged_layer(cfg, l) {
                k.push(None);
                v.push(None);
                k_row.push(0);
                v_row.push(0);
                continue;
            }
            k.push(Some(Arc::from(be.alloc(
                kv_side_bytes(k_fmt, pool_rows * kr),
                BufferUsage::KvCache,
            )?)));
            v.push(Some(Arc::from(be.alloc(
                kv_side_bytes(v_fmt, pool_rows * vr),
                BufferUsage::KvCache,
            )?)));
            k_row.push(kr);
            v_row.push(vr);
        }
        let mut ledger = PagePool::new(pages);
        // The zero page is never handed out and never freed: every unallocated table entry
        // points at it.
        let zero = ledger.alloc().expect("a pool has at least two pages");
        debug_assert_eq!(zero, 0);
        Ok(Arc::new(Self {
            k,
            v,
            k_row,
            v_row,
            k_fmt,
            v_fmt,
            pages: Mutex::new(ledger),
            pps,
            slot_rows,
            total_pages: pages,
        }))
    }

    /// Layer `l`'s K pool, or `None` when that layer keeps a private cache.
    pub(crate) fn k(&self, l: usize) -> Option<&Arc<dyn Buffer>> {
        self.k.get(l).and_then(|b| b.as_ref())
    }

    pub(crate) fn v(&self, l: usize) -> Option<&Arc<dyn Buffer>> {
        self.v.get(l).and_then(|b| b.as_ref())
    }

    /// The logical window (rows) one slot's table covers.
    pub(crate) fn slot_rows(&self) -> usize {
        self.slot_rows
    }

    /// Physical rows in every pool buffer (page 0 included).
    pub(crate) fn total_rows(&self) -> usize {
        self.total_pages * PAGE_ROWS
    }

    /// Entries a slot table needs: one per logical page plus the repeat of the last page that a
    /// tail chunk's one-page-ahead lookup reads.
    pub(crate) fn table_len(&self) -> usize {
        self.pps + 1
    }

    /// Physical pages in the budget, page 0 included.
    pub(crate) fn pages(&self) -> usize {
        self.total_pages
    }

    pub(crate) fn in_use_pages(&self) -> usize {
        self.pages.lock().expect("page-pool lock").in_use()
    }

    /// Pages a slot can actually own: the zero page is always referenced but never usable.
    pub(crate) fn used_pages(&self) -> usize {
        self.in_use_pages().saturating_sub(1)
    }

    pub(crate) fn free_pages(&self) -> usize {
        self.total_pages - self.in_use_pages()
    }

    pub(crate) fn ledger(&self) -> &Mutex<PagePool> {
        &self.pages
    }

    /// Copy `rows` rows of every pooled layer from a slot's linear buffers into the physical pages
    /// its map now holds.
    pub(crate) fn import_linear(
        &self,
        be: &dyn Backend,
        map: &SlotMap,
        rows: usize,
        src_k: &[Arc<dyn Buffer>],
        src_v: &[Arc<dyn Buffer>],
    ) -> AResult<()> {
        let rows = rows.min(self.slot_rows);
        for l in 0..self.k.len() {
            let Some(k) = self.k[l].as_ref() else {
                continue;
            };
            let v = self.v[l].as_ref().expect("pooled layer has both sides");
            copy_rows(
                be,
                src_k[l].as_ref(),
                &Linear,
                k.as_ref(),
                map,
                self.k_fmt,
                self.k_row[l],
                0..rows,
            )?;
            copy_rows(
                be,
                src_v[l].as_ref(),
                &Linear,
                v.as_ref(),
                map,
                self.v_fmt,
                self.v_row[l],
                0..rows,
            )?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for KvPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvPool")
            .field("pps", &self.pps)
            .field("slot_rows", &self.slot_rows)
            .field("pages", &self.total_pages)
            .field("in_use", &self.in_use_pages())
            .field(
                "pooled_layers",
                &self.k.iter().filter(|b| b.is_some()).count(),
            )
            .finish()
    }
}

/// One slot's logical-page mapping into a [`KvPool`]: `map[j]` is the physical page holding logical
/// page `j`, or 0 when the slot has not allocated it. Shared entries are released on drop, so a
/// recycled conversation slot gives its parent's pages back.
pub(crate) struct SlotMap {
    pool: Arc<KvPool>,
    map: Vec<u32>,
}

impl SlotMap {
    /// A slot with no pages yet.
    fn take(pool: &Arc<KvPool>) -> Self {
        Self {
            pool: Arc::clone(pool),
            map: Vec::new(),
        }
    }

    pub(crate) fn pool(&self) -> &Arc<KvPool> {
        &self.pool
    }

    /// Logical pages this slot has allocated (a prefix; the rest are zero page).
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    /// Physical page holding logical page `j`, 0 when unallocated.
    pub(crate) fn page(&self, j: usize) -> u32 {
        self.map.get(j).copied().unwrap_or(0)
    }

    /// Whether logical page `j` has this slot as its only owner, so a write needs no copy.
    pub(crate) fn is_private(&self, j: usize) -> bool {
        let p = self.page(j);
        p != 0 && self.pool.ledger().lock().expect("page-pool lock").refs(p) == 1
    }

    /// One physical page id per logical page of the window, plus the repeat of the last page.
    pub(crate) fn table_bytes(&self) -> Vec<u8> {
        let mut t: Vec<u32> = (0..self.pool.pps).map(|j| self.page(j)).collect();
        let last = t.last().copied().unwrap_or(0);
        t.push(last);
        let mut out = Vec::with_capacity(t.len() * 4);
        for p in t {
            out.extend_from_slice(&p.to_le_bytes());
        }
        out
    }

    /// Allocate private pages until the map covers `pages` logical pages.
    pub(crate) fn grow(&mut self, pages: usize) -> AResult<()> {
        let pages = pages.min(self.pool.pps);
        let mut ledger = self.pool.ledger().lock().expect("page-pool lock");
        while self.map.len() < pages {
            let Some(p) = ledger.alloc() else {
                return Err(anyhow::Error::new(KvPagesExhausted {
                    need: pages,
                    allocated: self.map.len(),
                    free: ledger.capacity() - ledger.in_use(),
                    total: ledger.capacity(),
                }));
            };
            self.map.push(p);
        }
        Ok(())
    }

    /// Share the first `pages` logical pages of `src`: the prefix both slots hold. Callers only
    /// pass whole pages; a prefix under one page is re-prefilled instead of shared.
    pub(crate) fn share_prefix(&mut self, src: &SlotMap, pages: usize) -> AResult<()> {
        let pages = pages.min(src.len()).min(self.pool.pps);
        self.grow(pages)?;
        let mut ledger = self.pool.ledger().lock().expect("page-pool lock");
        for j in 0..pages {
            let p = src.map[j];
            if self.map[j] == p {
                continue;
            }
            ledger.share(p);
            if self.map[j] != 0 {
                ledger.release(self.map[j]);
            }
            self.map[j] = p;
        }
        Ok(())
    }

    /// Release every page and start over empty. The rows are left alone — a caller that resets
    /// the cache re-prefills them.
    pub(crate) fn unshare_all(&mut self) {
        let mut ledger = self.pool.ledger().lock().expect("page-pool lock");
        for p in self.map.drain(..) {
            if p != 0 {
                ledger.release(p);
            }
        }
    }

    /// Split logical page `j` away from its sharers: copy the page into a fresh physical page,
    /// then release the shared reference. Device-to-device, per pooled layer and side.
    pub(crate) fn cow_page(&mut self, be: &dyn Backend, j: usize) -> AResult<()> {
        if j >= self.map.len() || self.is_private(j) {
            return Ok(());
        }
        let pool = Arc::clone(&self.pool);
        let src = self.map[j];
        let Some(own) = pool.ledger().lock().expect("page-pool lock").alloc() else {
            return Err(anyhow::Error::new(KvPagesExhausted {
                need: j + 1,
                allocated: self.map.len(),
                free: pool.free_pages(),
                total: pool.pages(),
            }));
        };
        for l in 0..pool.k.len() {
            let Some(k) = pool.k[l].as_ref() else {
                continue;
            };
            let kb = kv_fmt_bytes(pool.k_fmt, PAGE_ROWS * pool.k_row[l]);
            copy_page(be, k.as_ref(), src, own, kb)?;
            let v = pool.v[l].as_ref().expect("pooled layer has both sides");
            let vb = kv_fmt_bytes(pool.v_fmt, PAGE_ROWS * pool.v_row[l]);
            copy_page(be, v.as_ref(), src, own, vb)?;
        }
        if src != 0 {
            pool.ledger().lock().expect("page-pool lock").release(src);
        }
        self.map[j] = own;
        Ok(())
    }
}

/// Copy one page of `buf` from physical page `src` to physical page `dst`; `bytes` is that page's
/// size, so page `p` starts at `p * bytes`.
fn copy_page(be: &dyn Backend, buf: &dyn Buffer, src: u32, dst: u32, bytes: usize) -> AResult<()> {
    if bytes == 0 || src == dst {
        return Ok(());
    }
    be.copy_buffer_at(buf, src as usize * bytes, buf, dst as usize * bytes, bytes)
        .map_err(|e| anyhow!("{e}"))
}

impl Drop for SlotMap {
    fn drop(&mut self) {
        self.unshare_all();
    }
}

/// A slot's page map plus the table buffer every replayed attention dispatch reads it through.
pub(crate) struct PagedSlot {
    pub(crate) map: SlotMap,
    pub(crate) table: Box<dyn Buffer>,
    /// The table on the device is behind `map` (a release happened without a backend at hand).
    pub(crate) stale: bool,
}

impl PagedSlot {
    fn with_table(map: SlotMap, be: &dyn Backend) -> AResult<Self> {
        let bytes = map.pool().table_len() * 4;
        let table = be.alloc(bytes, BufferUsage::KvCache)?;
        let mut slot = Self {
            map,
            table,
            stale: false,
        };
        slot.upload_table(be)?;
        Ok(slot)
    }

    /// First pages of a slot: no pages allocated yet.
    pub(crate) fn new(pool: &Arc<KvPool>, be: &dyn Backend) -> AResult<Self> {
        Self::with_table(SlotMap::take(pool), be)
    }

    /// A fork: a free map whose first `pages` logical pages point at `src`'s pages.
    pub(crate) fn fork(
        pool: &Arc<KvPool>,
        be: &dyn Backend,
        src: &SlotMap,
        pages: usize,
    ) -> AResult<Self> {
        let mut map = SlotMap::take(pool);
        map.share_prefix(src, pages)?;
        Self::with_table(map, be)
    }

    /// Push the current mapping to the device. Called after every change to `map`.
    pub(crate) fn upload_table(&mut self, be: &dyn Backend) -> AResult<()> {
        let bytes = self.map.table_bytes();
        tracing::debug!(
            first = ?&bytes[..bytes.len().min(24)],
            "kv.paged: uploading a slot's page table"
        );
        be.upload(self.table.as_ref(), &bytes)
            .map_err(|e| anyhow!("{e}"))?;
        self.stale = false;
        Ok(())
    }

    /// The linear fallback bias. A paged slot resolves every row through its table, so this is 0.
    pub(crate) fn row_base(&self) -> usize {
        0
    }

    /// Allocate pages through `rows` and push the table if the map grew. Returns the page count
    /// now allocated. A refusal partway through leaves the map ahead of the device table, which
    /// `stale` records so the next forward uploads it.
    pub(crate) fn reserve_rows(&mut self, be: &dyn Backend, rows: usize) -> AResult<usize> {
        let need = rows.div_ceil(PAGE_ROWS).min(self.map.pool().pps);
        if self.map.len() >= need {
            return Ok(self.map.len());
        }
        self.stale = true;
        self.map.grow(need)?;
        self.upload_table(be)?;
        Ok(self.map.len())
    }

    /// Copy the rows a prefix has inside logical page `page` (fewer than `PAGE_ROWS` of them) from
    /// `src` to this slot's own page, so a shared prefix shorter than one page is still reused.
    pub(crate) fn copy_tail(
        &mut self,
        be: &dyn Backend,
        src: &SlotMap,
        page: usize,
        rows: usize,
    ) -> AResult<()> {
        let rows = rows.min(PAGE_ROWS);
        if rows == 0 || page >= self.map.pool().pps {
            return Ok(());
        }
        if self.map.len() <= page {
            self.stale = true;
            self.map.grow(page + 1)?;
        }
        let pool = self.map.pool();
        let span = page * PAGE_ROWS..page * PAGE_ROWS + rows;
        for l in 0..pool.k.len() {
            let Some(k) = pool.k[l].as_ref() else {
                continue;
            };
            let v = pool.v[l].as_ref().expect("pooled layer has both sides");
            let (k, v) = (k.as_ref(), v.as_ref());
            let (fk, fv) = (pool.k_fmt, pool.v_fmt);
            copy_rows(be, k, src, k, &self.map, fk, pool.k_row[l], span.clone())?;
            copy_rows(be, v, src, v, &self.map, fv, pool.v_row[l], span.clone())?;
        }
        self.upload_table(be)?;
        Ok(())
    }

    /// Drop every page without a device copy; the table is refreshed on the next forward.
    pub(crate) fn release_pages(&mut self) {
        self.map.unshare_all();
        self.stale = true;
    }

    /// Push a table left behind by [`PagedSlot::release_pages`].
    pub(crate) fn refresh(&mut self, be: &dyn Backend) -> AResult<()> {
        if self.stale {
            self.upload_table(be)?;
        }
        Ok(())
    }

    /// Take private copies of every page holding rows `[start, end)`, allocating the pages first.
    /// Returns how many pages it split, for logging.
    pub(crate) fn make_private(
        &mut self,
        be: &dyn Backend,
        start: usize,
        end: usize,
    ) -> AResult<usize> {
        if end == 0 {
            return Ok(0);
        }
        self.reserve_rows(be, end)?;
        let first_page = start / PAGE_ROWS;
        let last_page = (end - 1) / PAGE_ROWS;
        let mut split = 0;
        for j in first_page..=last_page {
            if !self.map.is_private(j) {
                self.map.cow_page(be, j)?;
                split += 1;
            }
        }
        if split > 0 {
            self.upload_table(be)?;
        }
        Ok(split)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

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

    fn cpu() -> infr_cpu::CpuBackend {
        infr_cpu::CpuBackend::new()
    }

    fn page_bytes(pool: &KvPool) -> usize {
        kv_fmt_bytes(pool.k_fmt, PAGE_ROWS * pool.k_row[0])
    }

    fn read_page(be: &dyn Backend, buf: &dyn Buffer, page: usize, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        be.download_at(buf, page * bytes, &mut out).unwrap();
        out
    }

    /// A TurboQuant KV side has no page-aware store or dequant prepass, so the pool must refuse
    /// rather than write rows the linear reads never find.
    #[test]
    fn a_turbo_kv_cache_refuses_paging() {
        let be = cpu();
        let c = cfg();
        for (k_fmt, v_fmt) in [
            (DType::Turbo3, DType::F16),
            (DType::F16, DType::Turbo4),
            (DType::Turbo2, DType::Turbo2),
        ] {
            let err = KvPool::new(&be, &c, 1024, k_fmt, v_fmt, 4)
                .expect_err("a turbo side must refuse the pool")
                .to_string();
            assert!(err.contains("Turbo"), "the error names the format: {err}");
        }
    }

    /// The whole point of the pool: a fork shares the parent's pages instead of copying them, a
    /// write to a shared page splits it first, and the parent's bytes survive the split.
    #[test]
    fn a_fork_shares_page_bytes_and_a_write_splits_them() {
        let be = cpu();
        let c = cfg();
        let pool = KvPool::new(&be, &c, 1024, DType::F16, DType::F16, 5).unwrap();
        assert_eq!(pool.pps, 2, "1024 rows of ctx over 512-row pages");
        assert_eq!(pool.slot_rows(), 1024);
        let bytes = page_bytes(&pool);
        assert_eq!(
            bytes,
            PAGE_ROWS * 2 * 16 * 2,
            "512 rows × (n_kv·head_dim) f16"
        );

        let mut src = SlotMap::take(&pool);
        src.grow(2).unwrap();
        let src_page = src.page(0);
        let k = pool.k(0).unwrap().clone();
        be.upload_at(k.as_ref(), src_page as usize * bytes, &vec![0xAAu8; bytes])
            .unwrap();

        let mut dst = SlotMap::take(&pool);
        dst.share_prefix(&src, 1).unwrap();
        assert_eq!(dst.page(0), src_page);
        assert_eq!(pool.pages.lock().unwrap().refs(src_page), 2);
        assert!(!dst.is_private(0));

        dst.cow_page(&be, 0).unwrap();
        assert_ne!(dst.page(0), src_page);
        assert_eq!(pool.pages.lock().unwrap().refs(src_page), 1);
        assert_eq!(
            read_page(&be, k.as_ref(), dst.page(0) as usize, bytes),
            vec![0xAAu8; bytes],
            "the split copied the shared page"
        );

        be.upload_at(k.as_ref(), src_page as usize * bytes, &[0x11u8; 64])
            .unwrap();
        assert_eq!(
            read_page(&be, k.as_ref(), dst.page(0) as usize, bytes)[..64],
            vec![0xAAu8; 64],
            "a parent write must not land in the child's page"
        );
        src.unshare_all();
        assert_eq!(pool.pages.lock().unwrap().refs(src_page), 0);
    }

    /// `make_private` splits exactly the pages covering `[start, end)` — the range a forward is
    /// about to write — and leaves a prefix shorter than one page shared.
    #[test]
    fn make_private_splits_only_the_rows_about_to_be_written() {
        let be = cpu();
        let c = cfg();
        let pool = KvPool::new(&be, &c, 1024, DType::F16, DType::F16, 5).unwrap();
        let mut src = SlotMap::take(&pool);
        src.grow(2).unwrap();
        let table = be
            .alloc(4 * pool.table_len(), BufferUsage::KvCache)
            .unwrap();
        let mut slot = PagedSlot {
            map: SlotMap::take(&pool),
            table,
            stale: false,
        };
        slot.map.share_prefix(&src, 2).unwrap();
        assert!(!slot.map.is_private(0) && !slot.map.is_private(1));

        let split = slot.make_private(&be, 600, 700).unwrap();
        assert_eq!(split, 1, "rows 600..700 live in page 1 only");
        assert!(!slot.map.is_private(0), "page 0 is still the parent's");
        assert!(slot.map.is_private(1));
    }

    /// A slot grows page by page; the pool refuses the page that does not fit instead of writing
    /// past its buffer, and a reset returns every page.
    #[test]
    fn grow_stops_at_the_budget_and_reset_returns_pages() {
        let be = cpu();
        let c = cfg();
        let pool = KvPool::new(&be, &c, 1024, DType::F16, DType::F16, 3).unwrap();
        let mut slot = PagedSlot::new(&pool, &be).unwrap();
        slot.reserve_rows(&be, 1024).unwrap();
        assert_eq!(slot.map.len(), 2);
        assert_eq!(pool.in_use_pages(), 3, "page 0 plus the slot's two");
        let mut other = PagedSlot::new(&pool, &be).unwrap();
        let err = other.reserve_rows(&be, 600).unwrap_err();
        assert!(
            err.downcast_ref::<KvPagesExhausted>().is_some(),
            "the refusal names the page budget: {err}"
        );
        slot.release_pages();
        assert_eq!(pool.in_use_pages(), 1, "only the zero page is left");
        other.reserve_rows(&be, 600).unwrap();
    }

    /// A refusal partway through a reservation leaves the map ahead of the device table, and a
    /// later request that fits the partial map must still see the table that map describes.
    #[test]
    fn a_partial_growth_leaves_the_table_stale() {
        let be = cpu();
        let c = cfg();
        let pool = KvPool::new(&be, &c, 1024, DType::F16, DType::F16, 4).unwrap();
        let mut first = PagedSlot::new(&pool, &be).unwrap();
        first.reserve_rows(&be, 1024).unwrap();
        assert!(!first.stale, "a whole reservation uploads its table");

        let mut second = PagedSlot::new(&pool, &be).unwrap();
        let err = second.reserve_rows(&be, 1024).unwrap_err();
        assert!(err.downcast_ref::<KvPagesExhausted>().is_some());
        assert_eq!(second.map.len(), 1, "one of the two pages was handed out");
        assert!(second.stale, "the device table does not describe the map");

        let mut got = vec![0u8; pool.table_len() * 4];
        second.refresh(&be).unwrap();
        assert!(!second.stale);
        be.download(second.table.as_ref(), &mut got).unwrap();
        assert_ne!(got[0..4], [0u8; 4], "the table now names the page");
    }

    /// A small and a large request share one budget: each holds only the pages it uses, the second
    /// cannot grow past what is left, and releasing both returns the pool to idle.
    #[test]
    fn a_small_and_a_large_request_share_one_page_budget() {
        let be = cpu();
        let c = cfg();
        let pool = KvPool::new(&be, &c, 1536, DType::F16, DType::F16, 1 + 5).unwrap();
        let mut small = PagedSlot::new(&pool, &be).unwrap();
        let mut large = PagedSlot::new(&pool, &be).unwrap();
        small.reserve_rows(&be, 700).unwrap();
        assert_eq!(small.map.len(), 2, "700 rows over 512-row pages");
        large.reserve_rows(&be, 1400).unwrap();
        assert_eq!(large.map.len(), 3);
        assert_eq!(pool.used_pages(), 5);
        assert_eq!(pool.free_pages(), 0);
        small.release_pages();
        assert_eq!(pool.used_pages(), 3);
        assert_eq!(pool.free_pages(), 2);
        large.release_pages();
        assert_eq!(pool.used_pages(), 0, "an idle pool is empty");
        assert_eq!(pool.free_pages(), 5);
    }

    /// LFM3 short-convolution layers hold recurrent state, not row-addressed KV.
    #[test]
    fn shortconv_layers_are_not_poolable() {
        let c = Config {
            n_layer: 2,
            n_head: 4,
            n_kv: 2,
            n_embd: 64,
            head_dim: 16,
            lfm3: true,
            shortconv_layers: vec![true, false],
            shortconv_l_cache: 4,
            ..Default::default()
        };
        assert!(c.is_shortconv_layer(0) && c.is_recurrent_layer(0));
        assert!(!paged_layer(&c, 0));
        assert!(paged_layer(&c, 1));
        assert!((0..c.n_layer)
            .filter(|&l| c.is_recurrent_layer(l))
            .all(|l| !paged_layer(&c, l)));
    }

    /// A prefix that ends inside a page must still be reused, not dropped: the tail rows are copied
    /// into the fork's own page while the pages before stay shared.
    #[test]
    fn a_prefix_shorter_than_a_page_is_copied() {
        let be = cpu();
        let c = cfg();
        let pool = KvPool::new(&be, &c, 1024, DType::F16, DType::F16, 5).unwrap();
        let bytes = page_bytes(&pool);
        let row = bytes / PAGE_ROWS;
        assert_eq!(row, 64, "n_kv·head_dim f16");

        let mut src = SlotMap::take(&pool);
        src.grow(1).unwrap();
        let src_page = src.page(0);
        let k = pool.k(0).unwrap().clone();
        be.upload_at(
            k.as_ref(),
            src_page as usize * bytes,
            &vec![0xAAu8; 7 * row],
        )
        .unwrap();

        let mut dst = PagedSlot::new(&pool, &be).unwrap();
        dst.copy_tail(&be, &src, 0, 7).unwrap();
        assert!(
            dst.map.is_private(0),
            "the tail lands in the fork's own page"
        );
        let page = read_page(&be, k.as_ref(), dst.map.page(0) as usize, bytes);
        assert!(page[..7 * row].iter().all(|&b| b == 0xAA));
        assert!(
            page[7 * row..].iter().all(|&b| b == 0),
            "nothing past the copied rows"
        );
    }
}
