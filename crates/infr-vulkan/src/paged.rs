//! Page-granular KV sharing (forward PagedAttention).
//!
//! The attention kernels already read a logical row through a page table
//! (`PageMap`); this module owns the physical side: which page holds which
//! logical block, how many slots point at it, and when a write has to break a
//! shared page. It is deliberately host-side and buffer-agnostic — the caller
//! copies bytes on the device when [`SlotPages::ensure_writable`] says a page
//! was unshared.

/// Rows per physical page. A decode chunk is at most 512 keys wide, so a chunk
/// spans at most two pages.
pub const PAGE_ROWS: usize = 512;

/// Physical page bookkeeping for one KV buffer (one layer, one side).
pub struct PagePool {
    refs: Vec<u32>,
    free: Vec<u32>,
    hand: u32,
}

impl PagePool {
    /// A pool of `pages` physical pages, none handed out.
    pub fn new(pages: usize) -> Self {
        Self {
            refs: vec![0; pages],
            free: Vec::new(),
            hand: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.refs.len()
    }

    /// Pages with at least one reference.
    pub fn in_use(&self) -> usize {
        self.refs.iter().filter(|r| **r > 0).count()
    }

    pub fn refs(&self, page: u32) -> u32 {
        self.refs.get(page as usize).copied().unwrap_or(0)
    }

    /// Take a page, or `None` when the pool is exhausted.
    pub fn alloc(&mut self) -> Option<u32> {
        let page = match self.free.pop() {
            Some(p) => p,
            None if (self.hand as usize) < self.refs.len() => {
                self.hand += 1;
                self.hand - 1
            }
            None => return None,
        };
        self.refs[page as usize] = 1;
        Some(page)
    }

    /// Add a reference, for a slot that is sharing this page's rows.
    pub fn share(&mut self, page: u32) -> u32 {
        let r = &mut self.refs[page as usize];
        *r += 1;
        *r
    }

    /// Drop a reference; the page returns to the pool at zero.
    pub fn release(&mut self, page: u32) {
        let r = &mut self.refs[page as usize];
        debug_assert!(*r > 0, "released page {page} twice");
        *r -= 1;
        if *r == 0 {
            self.free.push(page);
        }
    }
}

/// Where a write goes, and the page it must copy first when it was unshared.
pub struct Writable {
    pub page: u32,
    pub copy_from: Option<u32>,
}

/// One slot's logical pages: `pages[i]` holds logical rows `i*PAGE_ROWS..`.
#[derive(Default)]
pub struct SlotPages {
    pages: Vec<u32>,
}

impl SlotPages {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of pages the slot has mapped, i.e. capacity in whole pages.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    pub fn page(&self, logical: usize) -> Option<u32> {
        self.pages.get(logical).copied()
    }

    /// The page table the shader reads, plus a repeat of the last page because a
    /// tail chunk looks one page ahead.
    pub fn table(&self) -> Vec<u32> {
        let mut t = self.pages.clone();
        if let Some(&last) = t.last() {
            t.push(last);
        }
        t
    }

    /// Map a fresh page. `false` when the pool is exhausted.
    pub fn grow(&mut self, pool: &mut PagePool) -> bool {
        match pool.alloc() {
            Some(p) => {
                self.pages.push(p);
                true
            }
            None => false,
        }
    }

    /// Share every page of `src`, for a slot continuing the same prefix.
    pub fn fork(&mut self, pool: &mut PagePool, src: &SlotPages) {
        for &p in &src.pages {
            pool.share(p);
        }
        self.pages = src.pages.clone();
    }

    /// Make room to write into `logical`, unsharing the page if another slot
    /// points at it.
    pub fn ensure_writable(&mut self, pool: &mut PagePool, logical: usize) -> Option<Writable> {
        let old = self.page(logical)?;
        if pool.refs(old) == 1 {
            return Some(Writable {
                page: old,
                copy_from: None,
            });
        }
        let fresh = pool.alloc()?;
        pool.release(old);
        self.pages[logical] = fresh;
        Some(Writable {
            page: fresh,
            copy_from: Some(old),
        })
    }

    /// Drop every page past `pages`, releasing the references.
    pub fn truncate(&mut self, pool: &mut PagePool, pages: usize) {
        while self.pages.len() > pages {
            let p = self.pages.pop().expect("checked length");
            pool.release(p);
        }
    }

    /// Release the whole mapping.
    pub fn clear(&mut self, pool: &mut PagePool) {
        self.truncate(pool, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_returns_the_page_to_the_pool() {
        let mut pool = PagePool::new(2);
        let a = pool.alloc().unwrap();
        let b = pool.alloc().unwrap();
        assert_eq!(pool.in_use(), 2);
        assert!(pool.alloc().is_none());
        pool.release(a);
        assert_eq!(pool.alloc(), Some(a));
        assert_eq!(pool.in_use(), 2);
        assert_eq!(pool.capacity(), 2);
        pool.release(b);
    }

    #[test]
    fn fork_shares_pages_until_a_write_breaks_them() {
        let mut pool = PagePool::new(4);
        let mut a = SlotPages::new();
        assert!(a.grow(&mut pool));
        assert!(a.grow(&mut pool));
        let mut b = SlotPages::new();
        b.fork(&mut pool, &a);
        assert_eq!(pool.refs(a.page(0).unwrap()), 2);
        assert_eq!(pool.in_use(), 2);

        // A read-only fork costs no pages; the first write to a shared page does.
        let w = b.ensure_writable(&mut pool, 0).unwrap();
        assert_eq!(w.copy_from, Some(a.page(0).unwrap()));
        assert_ne!(w.page, a.page(0).unwrap());
        assert_eq!(pool.refs(a.page(0).unwrap()), 1);
        assert_eq!(pool.in_use(), 3);
        assert_eq!(
            b.ensure_writable(&mut pool, 0)
                .map(|w| (w.page, w.copy_from)),
            Some((w.page, None))
        );
    }

    #[test]
    fn truncate_releases_only_the_tail() {
        let mut pool = PagePool::new(4);
        let mut a = SlotPages::new();
        let mut b = SlotPages::new();
        a.grow(&mut pool);
        a.grow(&mut pool);
        b.fork(&mut pool, &a);
        b.truncate(&mut pool, 1);
        assert_eq!(b.len(), 1);
        assert_eq!(pool.refs(a.page(1).unwrap()), 1);
        assert_eq!(pool.in_use(), 2);
        b.clear(&mut pool);
        assert!(b.is_empty());
        assert_eq!(pool.in_use(), 2);
    }

    #[test]
    fn the_table_repeats_the_last_page() {
        let mut pool = PagePool::new(2);
        let mut a = SlotPages::new();
        a.grow(&mut pool);
        a.grow(&mut pool);
        let t = a.table();
        assert_eq!(t.len(), 3);
        assert_eq!(t[1], t[2]);
    }
}
