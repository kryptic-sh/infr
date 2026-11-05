//! Session-wide KV page pool: the physical half of forward PagedAttention in the seam.
//!
//! One [`KvPool`] serves every conversation slot of a session. Each slot owns a contiguous BLOCK
//! of pages inside every layer's pool buffer, and that block's first row is the slot's store bias
//! ([`infr_core::PageMap::row_base`]), so a store never consults the table. A forked slot starts by
//! SHARING its parent's prefix pages instead of copying them, and splits the sharing one page at a
//! time when it writes into shared rows.
//!
//! Page ids are token-granular and identical across layers: physical page `p` holds rows
//! `p*PAGE_ROWS..(p+1)*PAGE_ROWS` of EVERY pooled layer. That is what lets one table buffer per
//! SLOT (not per layer) drive every layer's reads. A slot's block mirrors one logical page space
//! one-to-one — logical page `j` lives at physical page `block*pps + j` while the slot is not
//! sharing it — so a copy-on-write always has a home and the bias stays a single offset.

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::DType;

use crate::Config;
use infr_vulkan::paged::{PagePool, PAGE_ROWS};

use crate::seam::{kv_fmt_bytes, kv_row_elems, kv_side_bytes};

/// Conversation slots one pool covers. Two is where sharing pays (the acceptance case is two
/// conversations over one long shared prefix) and it stays at what two linear slots already cost.
/// A further fork finds no block and fails, which the slot allocators already handle by recycling
/// their LRU slot.
pub(crate) const POOL_BLOCKS: usize = 2;

/// Whether layer `l`'s cache can live in the page pool: its reads go through the page table and its
/// stores through the block bias. SWA layers ring (position-addressed, never page-addressed), and
/// qwen35's DeltaNet layers and the MLA/lite architectures keep state the paged kernels do not
/// read.
pub(crate) fn paged_layer(cfg: &Config, l: usize) -> bool {
    if cfg.qwen35 && !cfg.is_qwen35_attn_layer(l) {
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
    /// Reference ledger: `refs[p]` counts the slots pointing at physical page `p`. A block's own
    /// pages start at 1 and never drop below it (see [`KvPool::new`]).
    pages: Mutex<PagePool>,
    /// Logical pages per slot block, and rows per slot block (the store bias stride).
    pps: usize,
    slot_rows: usize,
    blocks: usize,
    /// Blocks handed out so far — block 0 is the slot that existed when the pool was built, every
    /// later one is a fork.
    next_block: Mutex<usize>,
}

impl KvPool {
    /// Build a pool of `blocks` conversation slots with room for `max_ctx` rows each.
    pub(crate) fn new(
        be: &dyn Backend,
        cfg: &Config,
        max_ctx: usize,
        k_fmt: DType,
        v_fmt: DType,
        blocks: usize,
    ) -> AResult<Arc<Self>> {
        if let Some(why) = no_poolable_layer(cfg) {
            return Err(anyhow!("{why}"));
        }
        let blocks = blocks.max(1);
        let pps = max_ctx.div_ceil(PAGE_ROWS).max(1);
        let slot_rows = pps * PAGE_ROWS;
        let pool_rows = blocks * slot_rows;
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
        // A block's own pages hold one reference from birth: the ledger counts slots pointing at a
        // page, and every block page is always pointed at by its own slot.
        let mut pages = PagePool::new(blocks * pps);
        for p in 0..(blocks * pps) as u32 {
            pages.share(p);
        }
        Ok(Arc::new(Self {
            k,
            v,
            k_row,
            v_row,
            k_fmt,
            v_fmt,
            pages: Mutex::new(pages),
            pps,
            slot_rows,
            blocks,
            next_block: Mutex::new(0),
        }))
    }

    /// Layer `l`'s K pool, or `None` when that layer keeps a private cache.
    pub(crate) fn k(&self, l: usize) -> Option<&Arc<dyn Buffer>> {
        self.k.get(l).and_then(|b| b.as_ref())
    }

    pub(crate) fn v(&self, l: usize) -> Option<&Arc<dyn Buffer>> {
        self.v.get(l).and_then(|b| b.as_ref())
    }

    pub(crate) fn slot_rows(&self) -> usize {
        self.slot_rows
    }

    /// Entries a slot table needs: one per logical page plus the repeat of the last page that a
    /// tail chunk's one-page-ahead lookup reads.
    pub(crate) fn table_len(&self) -> usize {
        self.pps + 1
    }

    pub(crate) fn blocks(&self) -> usize {
        self.blocks
    }

    fn take_block(&self) -> Option<usize> {
        let mut n = self.next_block.lock().expect("page-pool block lock");
        if *n >= self.blocks {
            return None;
        }
        let b = *n;
        *n += 1;
        Some(b)
    }

    /// Copy `rows` rows of layer `l` from the slot's linear buffers into `block`'s rows.
    pub(crate) fn import_rows(
        &self,
        be: &dyn Backend,
        l: usize,
        block: usize,
        rows: usize,
        src_k: &dyn Buffer,
        src_v: &dyn Buffer,
    ) -> AResult<()> {
        let (k_row, v_row) = (self.k_row[l], self.v_row[l]);
        let k_off = kv_fmt_bytes(self.k_fmt, block * self.slot_rows * k_row);
        let v_off = kv_fmt_bytes(self.v_fmt, block * self.slot_rows * v_row);
        let k = self.k[l].as_ref().expect("pooled layer has both sides");
        let v = self.v[l].as_ref().expect("pooled layer has both sides");
        be.copy_buffer_at(
            src_k,
            0,
            k.as_ref(),
            k_off,
            kv_fmt_bytes(self.k_fmt, rows * k_row),
        )
        .map_err(|e| anyhow!("{e}"))?;
        be.copy_buffer_at(
            src_v,
            0,
            v.as_ref(),
            v_off,
            kv_fmt_bytes(self.v_fmt, rows * v_row),
        )
        .map_err(|e| anyhow!("{e}"))
    }
}

impl std::fmt::Debug for KvPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvPool")
            .field("pps", &self.pps)
            .field("slot_rows", &self.slot_rows)
            .field("blocks", &self.blocks)
            .field(
                "pooled_layers",
                &self.k.iter().filter(|b| b.is_some()).count(),
            )
            .finish()
    }
}

/// One slot's logical-page mapping into a [`KvPool`]: `map[j]` is the physical page holding logical
/// page `j`. Shared entries are released on drop, so a recycled conversation slot gives its
/// parent's pages back.
pub(crate) struct SlotMap {
    pool: Arc<KvPool>,
    block: usize,
    map: Vec<u32>,
}

impl SlotMap {
    /// Take a free block: every logical page on its own block page.
    fn take(pool: &Arc<KvPool>) -> AResult<Self> {
        let block = pool.take_block().ok_or_else(|| {
            anyhow!(
                "all {} page-pool blocks of this session are taken",
                pool.blocks()
            )
        })?;
        let base = block * pool.pps;
        Ok(Self {
            map: (0..pool.pps).map(|j| (base + j) as u32).collect(),
            pool: Arc::clone(pool),
            block,
        })
    }

    /// Physical block this map occupies inside the pool.
    pub(crate) fn block(&self) -> usize {
        self.block
    }

    pub(crate) fn pool(&self) -> &Arc<KvPool> {
        &self.pool
    }

    /// First row of this slot's block inside every layer's pool buffer (the store bias).
    pub(crate) fn row_base(&self) -> usize {
        self.block * self.pool.slot_rows
    }

    /// Whether logical page `j` sits in this slot's own block, so a write needs no copy.
    pub(crate) fn is_private(&self, j: usize) -> bool {
        self.map[j] as usize == self.block * self.pool.pps + j
    }

    /// One physical page id per logical page, plus the repeat of the last page.
    pub(crate) fn table_bytes(&self) -> Vec<u8> {
        let mut t = self.map.clone();
        if let Some(&last) = t.last() {
            t.push(last);
        }
        let mut out = Vec::with_capacity(t.len() * 4);
        for p in t {
            out.extend_from_slice(&p.to_le_bytes());
        }
        out
    }

    /// Share the first `pages` logical pages of `src`: the prefix both slots hold. Callers only
    /// pass whole pages; a prefix under one page is re-prefilled instead of shared.
    pub(crate) fn share_prefix(&mut self, src: &SlotMap, pages: usize) {
        let pages = pages.min(src.map.len()).min(self.map.len());
        let mut ledger = self.pool.pages.lock().expect("page-pool lock");
        for j in 0..pages {
            let p = src.map[j];
            if self.map[j] == p {
                continue;
            }
            ledger.share(p);
            if !self.is_private(j) {
                ledger.release(self.map[j]);
            }
            self.map[j] = p;
        }
    }

    /// Point every logical page back at this slot's own block and release shared references. The
    /// rows are left alone — a caller that resets the cache re-prefills them.
    pub(crate) fn unshare_all(&mut self) {
        let base = self.block * self.pool.pps;
        let mut ledger = self.pool.pages.lock().expect("page-pool lock");
        for (j, p) in self.map.iter_mut().enumerate() {
            if *p as usize != base + j {
                ledger.release(*p);
                *p = (base + j) as u32;
            }
        }
    }

    /// Split logical page `j` away from its sharers: copy the page into this slot's own block page,
    /// then release the shared reference. Device-to-device, per pooled layer and side.
    pub(crate) fn cow_page(&mut self, be: &dyn Backend, j: usize) -> AResult<()> {
        if j >= self.map.len() || self.is_private(j) {
            return Ok(());
        }
        let pool = Arc::clone(&self.pool);
        let own = (self.block * pool.pps + j) as u32;
        let src = self.map[j];
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
        pool.pages.lock().expect("page-pool lock").release(src);
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
        let slot = Self {
            map,
            table,
            stale: false,
        };
        slot.upload_table(be)?;
        Ok(slot)
    }

    /// First pages of a slot: a free block, all pages private.
    pub(crate) fn new(pool: &Arc<KvPool>, be: &dyn Backend) -> AResult<Self> {
        Self::with_table(SlotMap::take(pool)?, be)
    }

    /// A fork: a free block whose first `pages` logical pages point at `src`'s pages.
    pub(crate) fn fork(
        pool: &Arc<KvPool>,
        be: &dyn Backend,
        src: &SlotMap,
        pages: usize,
    ) -> AResult<Self> {
        let mut map = SlotMap::take(pool)?;
        map.share_prefix(src, pages);
        Self::with_table(map, be)
    }

    /// Push the current mapping to the device. Called after every change to `map`.
    pub(crate) fn upload_table(&self, be: &dyn Backend) -> AResult<()> {
        be.upload(self.table.as_ref(), &self.map.table_bytes())
            .map_err(|e| anyhow!("{e}"))
    }

    pub(crate) fn row_base(&self) -> usize {
        self.map.row_base()
    }

    /// Drop every shared page without a device copy; the table is refreshed on the next forward.
    pub(crate) fn release_pages(&mut self) {
        self.map.unshare_all();
        self.stale = true;
    }

    /// Push a table left behind by [`PagedSlot::release_pages`].
    pub(crate) fn refresh(&mut self, be: &dyn Backend) -> AResult<()> {
        if self.stale {
            self.upload_table(be)?;
            self.stale = false;
        }
        Ok(())
    }

    /// Take private copies of every page holding rows `[start, end)`. Returns how many pages it
    /// split, for logging.
    pub(crate) fn make_private(
        &mut self,
        be: &dyn Backend,
        start: usize,
        end: usize,
    ) -> AResult<usize> {
        let last_page = (self.map.map.len() - 1).min(end.saturating_sub(1) / PAGE_ROWS);
        let first_page = (start / PAGE_ROWS).min(last_page);
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
    use infr_core::backend::Backend;

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
        crate::seam::kv_fmt_bytes(pool.k_fmt, PAGE_ROWS * pool.k_row[0])
    }

    fn read_page(be: &dyn Backend, buf: &dyn Buffer, page: usize, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        be.download_at(buf, page * bytes, &mut out).unwrap();
        out
    }

    /// The whole point of the pool: a fork shares the parent's pages instead of copying them, a
    /// write to a shared page splits it first, and the parent's bytes survive the split.
    #[test]
    fn a_fork_shares_page_bytes_and_a_write_splits_them() {
        let be = cpu();
        let c = cfg();
        let pool = KvPool::new(&be, &c, 1024, DType::F16, DType::F16, 2).unwrap();
        assert_eq!(pool.pps, 2, "1024 rows of ctx over 512-row pages");
        assert_eq!(pool.slot_rows(), 1024);
        let bytes = page_bytes(&pool);
        assert_eq!(
            bytes,
            PAGE_ROWS * 2 * 16 * 2,
            "512 rows × (n_kv·head_dim) f16"
        );

        let mut src = SlotMap::take(&pool).unwrap();
        assert_eq!(src.row_base(), 0);
        let k = pool.k(0).unwrap().clone();
        be.upload_at(k.as_ref(), 0, &vec![0xAAu8; bytes]).unwrap();

        let mut dst = SlotMap::take(&pool).unwrap();
        assert_eq!(dst.row_base(), 1024, "second block starts at row 1024");
        dst.share_prefix(&src, 1);
        assert_eq!(dst.table_bytes()[..8], [0u8, 0, 0, 0, 3, 0, 0, 0]);
        assert_eq!(pool.pages.lock().unwrap().refs(0), 2);

        dst.cow_page(&be, 0).unwrap();
        assert_eq!(dst.table_bytes()[..8], [2u8, 0, 0, 0, 3, 0, 0, 0]);
        assert_eq!(pool.pages.lock().unwrap().refs(0), 1);
        assert_eq!(
            read_page(&be, k.as_ref(), 0, bytes),
            vec![0xAAu8; bytes],
            "the split copied the shared page"
        );
        assert_eq!(
            read_page(&be, k.as_ref(), 2, bytes),
            vec![0xAAu8; bytes],
            "…into the slot's own page"
        );

        be.upload_at(
            k.as_ref(),
            src.row_base() * (bytes / PAGE_ROWS),
            &[0x11u8; 64],
        )
        .unwrap();
        assert_eq!(
            read_page(&be, k.as_ref(), 2, bytes)[..64],
            vec![0xAAu8; 64],
            "a parent write must not land in the child's page"
        );
        src.unshare_all();
        assert_eq!(pool.pages.lock().unwrap().refs(2), 1);
    }

    /// `make_private` splits exactly the pages covering `[start, end)` — the range a forward is
    /// about to write — and leaves a prefix shorter than one page shared.
    #[test]
    fn make_private_splits_only_the_rows_about_to_be_written() {
        let be = cpu();
        let c = cfg();
        let pool = KvPool::new(&be, &c, 1024, DType::F16, DType::F16, 2).unwrap();
        let src = SlotMap::take(&pool).unwrap();
        let table = be
            .alloc(4 * pool.table_len(), BufferUsage::KvCache)
            .unwrap();
        let mut slot = PagedSlot {
            map: SlotMap::take(&pool).unwrap(),
            table,
            stale: false,
        };
        slot.map.share_prefix(&src, 2);
        assert!(!slot.map.is_private(0) && !slot.map.is_private(1));

        let split = slot.make_private(&be, 600, 700).unwrap();
        assert_eq!(split, 1, "rows 600..700 live in page 1 only");
        assert!(!slot.map.is_private(0), "page 0 is still the parent's");
        assert!(slot.map.is_private(1));
    }
}
