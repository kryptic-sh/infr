//! Cross-conversation KV store, keyed in fixed-size token blocks.
//!
//! A prompt's K/V rows never change once written, so the rows one request materialized can seed
//! any later request that starts with the same tokens — a new conversation, an evicted slot, or a
//! different chat on the same model. The slot pool already reuses KV between its own slots
//! (`ParallelSeam::checkout` seeds from a free slot); this store is the same idea with the rows in
//! a tier of stores, so the reuse survives eviction and outlives the conversation that produced it.
//!
//! Rows are stored per 128-token block, chained: a block's id hashes its predecessor's id with its
//! own tokens, so a block only matches at the position it was written at (rope makes rows
//! position-bound) and ten conversations sharing a system prompt keep ONE copy of it.
//!
//! The payoff is the prefill it skips: 8k of KV uploads over PCIe in about 0.12 s, while
//! re-prefilling the same tokens costs tens of seconds.

use crate::Config;
use anyhow::{anyhow, Result};
use infr_core::backend::Backend;
use infr_core::DType;
use std::collections::HashMap;
use std::sync::Arc;

use super::{
    kv_tier::{
        decode_meta, encode_meta, Bounded, BudgetedStore, DeviceKv, Fnv, KvRows, KvStore, RedbKv,
        RedisKv, Tiered, FORMATS_META_ID,
    },
    pool::{PagedSlot, SlotMap},
    weights::{RecurrentSnap, SeamKv},
};
use infr_vulkan::paged::PAGE_ROWS;

/// Tokens per stored block. Smaller blocks share more (a 600-token system prompt with two
/// different tails stores each half once) at the price of one download per new block.
pub(crate) const BLOCK_TOKENS: usize = 128;

/// Below this many shared tokens the upload costs more than re-prefilling them.
pub(crate) const MIN_PREFIX_TOKENS: usize = 256;

/// Whole blocks in one KV page: a retained page covers exactly this many chained blocks.
const BLOCKS_PER_PAGE: usize = PAGE_ROWS / BLOCK_TOKENS;
const _: () = assert!(PAGE_ROWS.is_multiple_of(BLOCK_TOKENS));

/// Rows a capture keeps: every row the slot wrote, so the reply is reusable too. A recurrent model
/// resumes only from a snapshot, which prefill takes at prompt boundaries, so its reply rows would
/// never be read back.
pub(crate) fn capture_upto(cfg: &Config, kv: &SeamKv, prompt_len: usize) -> usize {
    if cfg.has_recurrent_state() {
        prompt_len
    } else {
        kv.cached_len()
    }
}

/// Chain id of a block: the previous block's id mixed with this block's tokens.
///
/// FNV-1a, not `DefaultHasher`: the ids are written to a tier, so they must not change with the
/// toolchain. Every lookup also compares the token ids, so a collision costs a miss, and a
/// capture under a colliding id replaces the older block.
fn block_id(prev: u64, tokens: &[u32]) -> u64 {
    let mut h = Fnv::new();
    h.bytes(&prev.to_le_bytes());
    for t in tokens {
        h.bytes(&t.to_le_bytes());
    }
    h.0
}

/// Identity of the model a store's blocks belong to: the geometry that shapes every row, plus the
/// context and vocabulary. Blocks under another key are never loaded.
pub(crate) fn store_key(cfg: &Config) -> u64 {
    let mut h = Fnv::new();
    for v in [
        cfg.n_layer,
        cfg.n_embd,
        cfg.n_head,
        cfg.n_kv,
        cfg.head_dim,
        cfg.rope_dim,
        cfg.n_ctx_train,
        cfg.vocab,
        cfg.n_ff,
    ] {
        h.bytes(&(v as u64).to_le_bytes());
    }
    h.0
}

/// A KV dtype's tag in the formats blob.
fn fmt_tag(d: DType) -> Option<u8> {
    Some(match d {
        DType::F32 => 1,
        DType::F16 => 2,
        DType::Bf16 => 3,
        DType::Q8_0 => 4,
        DType::Q4_0 => 5,
        DType::Q4_1 => 6,
        DType::Q5_0 => 7,
        DType::Q5_1 => 8,
        DType::Iq4Nl => 9,
        DType::Turbo2 => 10,
        DType::Turbo3 => 11,
        DType::Turbo4 => 12,
        _ => return None,
    })
}

fn fmt_from_tag(tag: u8) -> Option<DType> {
    Some(match tag {
        1 => DType::F32,
        2 => DType::F16,
        3 => DType::Bf16,
        4 => DType::Q8_0,
        5 => DType::Q4_0,
        6 => DType::Q4_1,
        7 => DType::Q5_0,
        8 => DType::Q5_1,
        9 => DType::Iq4Nl,
        10 => DType::Turbo2,
        11 => DType::Turbo3,
        12 => DType::Turbo4,
        _ => return None,
    })
}

/// The formats blob: the KV layouts every stored block was captured with.
fn encode_formats(k_fmt: DType, v_fmt: DType) -> Option<Vec<u8>> {
    let mut blob = vec![1, fmt_tag(k_fmt)?, fmt_tag(v_fmt)?];
    let mut h = Fnv::new();
    h.bytes(&blob);
    blob.extend_from_slice(&h.0.to_le_bytes());
    Some(blob)
}

fn decode_formats(blob: &[u8]) -> Option<(DType, DType)> {
    if blob.len() != 11 || blob[0] != 1 {
        return None;
    }
    let mut h = Fnv::new();
    h.bytes(&blob[..3]);
    if blob[3..] != h.0.to_le_bytes() {
        return None;
    }
    Some((fmt_from_tag(blob[1])?, fmt_from_tag(blob[2])?))
}

/// The index entry of one stored block: `tokens.len()` token ids whose rows are extent `id`.
///
/// The length is `BLOCK_TOKENS` except for the tail of a captured prefix, which may be short; a
/// short block only ever matches at the end of a prompt.
struct Entry {
    tokens: Vec<u32>,
    /// Rows plus snapshot, what `stats` reports.
    bytes: usize,
    /// A recurrent-state snapshot rides at the block's end inside the same extent.
    snap: bool,
    /// The store holds the block's extent. `false` for a block only a resident page map holds.
    stored: bool,
}

/// A parked conversation's pages, kept in the pool instead of copied out: the first
/// `map.len()` whole pages of the chain `blocks` names. A restore adopts them by reference.
struct Resident {
    map: SlotMap,
    /// Chained ids of the full blocks the pages hold, from token 0.
    blocks: Vec<u64>,
    tokens: Vec<u32>,
    /// LRU stamp; the oldest resident is spilled first.
    tick: u64,
}

/// The prefix cache: a token index over content-addressed block extents in `S`.
///
/// Rows are kept per 128-token block, chained: a block's id hashes its predecessor's id with its
/// own tokens, so a block only matches at the position it was written at (rope makes rows
/// position-bound) and ten conversations sharing a system prompt keep one copy of it. A block's
/// rows and its recurrent snapshot are one extent `id`, so an eviction moves or drops both.
pub(crate) struct BlockCache<S: KvStore> {
    store: S,
    index: HashMap<u64, Entry>,
    /// Whether `kv.tiers` configured a cache at all.
    enabled: bool,
    /// The KV layouts the blocks were captured with; a dtype change invalidates every block.
    k_fmt: DType,
    v_fmt: DType,
    /// Parked page maps, held in the pool until a restore adopts them or pressure spills them.
    resident: Vec<Resident>,
    tick: u64,
    /// Restores that read rows from the store, and the tokens they read.
    pub(crate) restores: u64,
    pub(crate) restored_tokens: u64,
    pub(crate) captures: u64,
    /// Restores that adopted a resident's pages, and the tokens those pages held.
    pub(crate) resident_hits: u64,
    pub(crate) resident_tokens: u64,
    /// Residents given back to the pool under page pressure.
    pub(crate) resident_spills: u64,
}

impl<S: KvStore> BlockCache<S> {
    /// Build a cache over `store`. A disabled cache returns immediately from every call, so the
    /// engine pays nothing for a feature nobody asked for. Loading reads the index only, never a
    /// block's bytes.
    pub(crate) fn new(store: S, enabled: bool) -> Self {
        let mut cache = Self {
            store,
            index: HashMap::new(),
            enabled,
            k_fmt: DType::F16,
            v_fmt: DType::F16,
            resident: Vec::new(),
            tick: 0,
            restores: 0,
            restored_tokens: 0,
            captures: 0,
            resident_hits: 0,
            resident_tokens: 0,
            resident_spills: 0,
        };
        if enabled {
            cache.load();
        }
        cache
    }

    /// Fill the index from the store's extent ids and their metadata. A missing or unreadable
    /// formats blob leaves the cache empty; it never fails startup.
    fn load(&mut self) {
        let formats = match self.store.meta(FORMATS_META_ID) {
            Ok(Some(blob)) => blob,
            Ok(None) => return,
            Err(e) => {
                tracing::warn!(error = %e, "kv store: reading the saved formats failed");
                return;
            }
        };
        let Some((k_fmt, v_fmt)) = decode_formats(&formats) else {
            tracing::warn!("kv store: the saved formats are unreadable; starting empty");
            return;
        };
        self.k_fmt = k_fmt;
        self.v_fmt = v_fmt;
        let ids = match self.store.extents() {
            Ok(ids) => ids,
            Err(e) => {
                tracing::warn!(error = %e, "kv store: listing the saved extents failed");
                return;
            }
        };
        let ids: Vec<u64> = ids
            .into_iter()
            .filter(|id| *id != FORMATS_META_ID)
            .collect();
        let metas = match self.store.metas(&ids) {
            Ok(metas) => metas,
            Err(e) => {
                tracing::warn!(error = %e, "kv store: reading the saved index entries failed");
                return;
            }
        };
        for (id, meta) in ids.into_iter().zip(metas) {
            if let Some(blob) = meta {
                match decode_meta(&blob) {
                    Ok((tokens, snap, bytes)) => {
                        self.index.insert(
                            id,
                            Entry {
                                tokens,
                                bytes,
                                snap,
                                stored: true,
                            },
                        );
                    }
                    Err(e) => {
                        tracing::warn!(id, error = %e, "kv store: dropped a block's bad index entry")
                    }
                }
            }
        }
        tracing::info!(
            blocks = self.index.len(),
            "kv store: loaded the saved prefix index"
        );
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// `(blocks, bytes)` currently resident, for the capture log.
    pub(crate) fn stats(&self) -> (usize, usize) {
        (self.index.len(), self.index.values().map(|e| e.bytes).sum())
    }

    /// Store block `id`'s rows and index it, replacing any block under the same id.
    fn insert(&mut self, id: u64, tokens: Vec<u32>, rows: KvRows) -> Result<()> {
        let bytes = rows.bytes();
        let snap = rows.snap.is_some();
        self.store.put(id, 0, rows)?;
        self.index.insert(
            id,
            Entry {
                tokens,
                bytes,
                snap,
                stored: true,
            },
        );
        self.write_meta(id)
    }

    /// Rewrite the block's extent with `snap` at its end, so the rows and their state evict
    /// together.
    fn attach(&mut self, id: u64, snap: Arc<RecurrentSnap>) -> Result<bool> {
        let Some(len) = self.index.get(&id).map(|e| e.tokens.len()) else {
            return Ok(false);
        };
        let mut rows = KvRows::default();
        if !self.store.get(id, 0..len, &mut rows)? {
            self.drop_entry(id);
            return Ok(false);
        }
        rows.snap = Some(snap);
        let bytes = rows.bytes();
        self.store.put(id, 0, rows)?;
        let entry = self.index.get_mut(&id).expect("an indexed block");
        entry.bytes = bytes;
        entry.snap = true;
        self.write_meta(id)?;
        Ok(true)
    }

    /// Persist the block's index entry beside its extent.
    fn write_meta(&mut self, id: u64) -> Result<()> {
        let Some(entry) = self.index.get(&id) else {
            return Ok(());
        };
        let blob = encode_meta(&entry.tokens, entry.snap, entry.bytes);
        self.store.set_meta(id, &blob)
    }

    fn drop_entry(&mut self, id: u64) {
        let Some(entry) = self.index.remove(&id) else {
            return;
        };
        if !entry.stored {
            return;
        }
        if let Err(e) = self.store.remove(id) {
            tracing::warn!(error = %e, "kv store: dropping a block's extent failed");
        }
    }

    fn clear(&mut self) {
        self.resident.clear();
        let ids: Vec<u64> = self.index.keys().copied().collect();
        for id in ids {
            self.drop_entry(id);
        }
    }

    /// Whether the store holds block `id` with exactly `tokens`.
    fn find(&self, id: u64, tokens: &[u32]) -> bool {
        self.index
            .get(&id)
            .is_some_and(|e| e.stored && e.tokens == tokens)
    }

    /// Whether a resident page map holds block `id`.
    fn resident_covers(&self, id: u64) -> bool {
        self.resident.iter().any(|r| r.blocks.contains(&id))
    }

    /// Walk `prompt` block by block; return the tokens covered by the leading stored run and the
    /// ids of the blocks that cover them, with their use stamped.
    fn matched(&mut self, prompt: &[u32]) -> (usize, Vec<u64>) {
        let mut id = 0u64;
        let mut n = 0usize;
        let mut hits = Vec::new();
        while n < prompt.len() {
            let len = BLOCK_TOKENS.min(prompt.len() - n);
            let block = &prompt[n..n + len];
            id = block_id(id, block);
            let stored = self.store.contains(id);
            if !stored && !self.resident_covers(id) {
                self.index.remove(&id);
                break;
            }
            let Some(entry) = self.index.get_mut(&id) else {
                break;
            };
            // The store evicted it while a resident kept its rows: the next capture stores it again.
            entry.stored &= stored;
            if entry.tokens != block {
                break;
            }
            hits.push(id);
            n += len;
            if len < BLOCK_TOKENS {
                break;
            }
        }
        (n, hits)
    }

    /// Copy the longest stored run of blocks into `kv` (a device upload per layer per run of
    /// present rows) and mark those tokens materialized. Returns the number of tokens restored.
    ///
    /// A sliding-window ring is written at each block's own ring position (`start % rows`), so a
    /// prefix longer than the ring still seeds it: the newest blocks overwrite the oldest, which
    /// is exactly the window attention will read. Only a destination whose own cache already
    /// wrapped is refused — its rows no longer describe its `cached`.
    pub(crate) fn restore(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        kv: &mut SeamKv,
        prompt: &[u32],
    ) -> Result<usize> {
        if !self.enabled() && self.resident.is_empty() {
            return Ok(0);
        }
        let (mut p, hits) = self.matched(prompt);
        if p < MIN_PREFIX_TOKENS || self.k_fmt != kv.k_fmt || self.v_fmt != kv.v_fmt {
            return Ok(0);
        }
        p = p.min(kv.max_ctx);
        if kv
            .window_ring_rows(be, cfg)
            .is_some_and(|rows| kv.cached.len() > rows)
        {
            return Ok(0);
        }
        let len = |id: &u64| self.index[id].tokens.len();
        let mut keep = Vec::new();
        let mut n = 0usize;
        for id in hits {
            if n + len(&id) > p {
                break;
            }
            n += len(&id);
            keep.push(id);
        }
        // A recurrent model's rows are useless without the state they were written beside: only a
        // block carrying a snapshot can seed it, and everything after that block must be prefilled.
        if cfg.has_recurrent_state() {
            let Some(at) = keep.iter().rposition(|id| self.index[id].snap) else {
                return Ok(0);
            };
            keep.truncate(at + 1);
            n = keep.iter().map(len).sum();
        }
        if n < MIN_PREFIX_TOKENS {
            return Ok(0);
        }
        // Whole pages a resident already holds are adopted in place; only the blocks past them
        // are read from the store.
        let adopt = self.resident_for(kv, prompt, &keep);
        let first = adopt.map_or(0, |(_, pages)| pages * PAGE_ROWS);
        let tail = adopt.map_or(0, |(_, pages)| pages * BLOCKS_PER_PAGE);
        let mut rows = KvRows::default();
        let mut n = first;
        for id in &keep[tail..] {
            let len = self.index[id].tokens.len();
            match self.store.get(*id, 0..len, &mut rows) {
                Ok(true) => n += len,
                // Adopted pages are a prefix on their own; the rest prefills.
                Ok(false) if adopt.is_some() => break,
                Ok(false) => return Ok(0),
                Err(e) => {
                    // A damaged extent is a cache miss, never a failed request.
                    tracing::warn!(id, error = %e, "kv store: dropping an unreadable block");
                    self.drop_entry(*id);
                    return Ok(0);
                }
            }
        }
        // Every row attention will read must be there: a window layer keeps only the newest rows
        // it has, and a restore that cannot fill that window must re-prefill instead.
        for l in 0..cfg.n_layer {
            if cfg.is_recurrent_layer(l) {
                continue;
            }
            let need = kv.row_layout(be, cfg, l).oldest_resident(n).max(first)..n;
            let held = need.start - first..need.end - first;
            if !need.is_empty() && !rows.layers.get(l).is_some_and(|r| r.covers(held)) {
                tracing::debug!(layer = l, need = ?need, "kv store: the stored blocks miss the window");
                return Ok(0);
            }
        }
        if let (Some((i, pages)), Some(p)) = (adopt, kv.paged.as_mut()) {
            p.map.share_prefix(&self.resident[i].map, pages)?;
            p.stale = true;
            self.tick += 1;
            self.resident[i].tick = self.tick;
            // The tail's pages may be the ones another resident holds.
            self.spill(page_deficit(p, first, n));
        }
        if n > first {
            if let Err(e) = upload_prefix(be, cfg, kv, first, n, rows) {
                // The map holds another conversation's rows now: give the pages back rather than
                // leave the slot claiming a prefix it no longer has.
                kv.reset();
                return Err(e);
            }
            self.restores += 1;
            self.restored_tokens += (n - first) as u64;
        }
        if adopt.is_some() {
            self.resident_hits += 1;
            self.resident_tokens += first as u64;
            tracing::debug!(
                adopted = first,
                read = n - first,
                "kv store: resumed from resident pages"
            );
        }
        kv.written_end = n;
        kv.cached = prompt[..n].to_vec();
        Ok(n)
    }

    /// The resident holding the most whole pages of `keep`'s chain, and that page count. Only a
    /// paged slot on the same pool can adopt.
    fn resident_for(&self, kv: &SeamKv, prompt: &[u32], keep: &[u64]) -> Option<(usize, usize)> {
        let pool = kv.paged.as_ref()?.map.pool();
        self.resident
            .iter()
            .enumerate()
            .filter(|(_, r)| Arc::ptr_eq(r.map.pool(), pool))
            .map(|(i, r)| {
                let blocks = r.blocks.iter().zip(keep).take_while(|(a, b)| a == b);
                (i, (blocks.count() / BLOCKS_PER_PAGE).min(r.map.len()))
            })
            .filter(|&(i, pages)| {
                let rows = pages * PAGE_ROWS;
                pages > 0 && self.resident[i].tokens.get(..rows) == prompt.get(..rows)
            })
            .max_by_key(|&(_, pages)| pages)
    }

    /// Keep a parked slot's pages in the pool instead of copying them out: its map moves into a
    /// resident entry over the whole pages its tokens fill, and the slot is left with none.
    /// `false` when the slot cannot be retained — linear, a layer outside the pool (a window
    /// ring, a recurrent state, a compressed cache), or under one page of tokens — so the caller
    /// resets it as before.
    pub(crate) fn retain(&mut self, cfg: &Config, kv: &mut SeamKv) -> bool {
        let Some(p) = kv.paged.as_mut() else {
            return false;
        };
        // A page holds every pooled layer's rows; a layer outside the pool would come back empty.
        if cfg.has_recurrent_state() || !(0..cfg.n_layer).all(|l| super::paged_layer(cfg, l)) {
            return false;
        }
        let pages = (kv.cached.len().min(kv.max_ctx) / PAGE_ROWS).min(p.map.len());
        if pages == 0 {
            return false;
        }
        if self.k_fmt != kv.k_fmt || self.v_fmt != kv.v_fmt {
            self.clear();
            self.k_fmt = kv.k_fmt;
            self.v_fmt = kv.v_fmt;
        }
        let tokens = kv.cached[..pages * PAGE_ROWS].to_vec();
        let mut blocks = Vec::with_capacity(pages * BLOCKS_PER_PAGE);
        let mut id = 0u64;
        for block in tokens.chunks(BLOCK_TOKENS) {
            id = block_id(id, block);
            match self.index.get(&id) {
                Some(e) if e.tokens == block => {}
                // Colliding ids: the stored block keeps the id and these pages are not kept.
                Some(_) => return false,
                None => {
                    self.index.insert(
                        id,
                        Entry {
                            tokens: block.to_vec(),
                            bytes: 0,
                            snap: false,
                            stored: false,
                        },
                    );
                }
            }
            blocks.push(id);
        }
        let mut map = p.take_map();
        map.truncate(pages);
        // A shorter resident on the same chain holds nothing this one does not.
        self.resident
            .retain(|r| !(r.blocks.len() <= blocks.len() && blocks.starts_with(&r.blocks)));
        self.tick += 1;
        self.resident.push(Resident {
            map,
            blocks,
            tokens,
            tick: self.tick,
        });
        tracing::debug!(
            pages,
            residents = self.resident.len(),
            "kv store: kept a parked slot's pages"
        );
        true
    }

    /// Give residents' pages back to the pool, least recently used first, until `pages` more are
    /// free or none is left. A block only a resident held is forgotten with it. Returns how many
    /// residents went.
    pub(crate) fn spill(&mut self, pages: usize) -> usize {
        let Some(pool) = self.resident.first().map(|r| Arc::clone(r.map.pool())) else {
            return 0;
        };
        let want = pool.free_pages() + pages;
        let mut spilled = 0;
        while pool.free_pages() < want {
            let Some(i) = (0..self.resident.len()).min_by_key(|&i| self.resident[i].tick) else {
                break;
            };
            let gone = self.resident.swap_remove(i);
            for id in &gone.blocks {
                let only_here = self.index.get(id).is_some_and(|e| !e.stored);
                if only_here && !self.resident_covers(*id) {
                    self.index.remove(id);
                }
            }
            drop(gone);
            spilled += 1;
        }
        self.resident_spills += spilled as u64;
        if spilled > 0 {
            tracing::debug!(
                spilled,
                free = pool.free_pages(),
                "kv store: spilled resident pages"
            );
        }
        spilled
    }

    /// Spill residents until `kv` can hold `rows` rows: the pages its run still has to allocate
    /// for rows past what it already cached, less the pages the pool has free. Returns how many
    /// residents went.
    pub(crate) fn make_room(&mut self, kv: &SeamKv, rows: usize) -> usize {
        let Some(p) = kv.paged.as_ref() else {
            return 0;
        };
        if self.resident.is_empty() {
            return 0;
        }
        self.spill(page_deficit(p, kv.cached.len(), rows.min(kv.max_ctx)))
    }

    /// Capture `kv`'s first `upto` rows as reusable blocks. Returns the token count kept.
    ///
    /// Blocks already stored are skipped, so a conversation that extends its own prefix keeps only
    /// the new suffix. A recurrent layer has no per-position rows: its state rides along as a
    /// snapshot on the block that ends where the state was captured, and a restore needs one.
    pub(crate) fn capture(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        kv: &SeamKv,
        upto: usize,
    ) -> Result<usize> {
        if !self.enabled() {
            return Ok(0);
        }
        let p = upto.min(kv.cached.len()).min(kv.max_ctx);
        tracing::debug!(
            upto,
            cached = kv.cached.len(),
            p,
            snaps = kv.recurrent_snaps.len(),
            "kv store: capture plan"
        );
        if p < MIN_PREFIX_TOKENS {
            return Ok(0);
        }
        if self.k_fmt != kv.k_fmt || self.v_fmt != kv.v_fmt {
            self.clear();
            self.k_fmt = kv.k_fmt;
            self.v_fmt = kv.v_fmt;
        }
        if let Some(blob) = encode_formats(self.k_fmt, self.v_fmt) {
            self.store.set_meta(FORMATS_META_ID, &blob)?;
        }
        let tokens = &kv.cached[..p];
        let mut id = 0u64;
        let mut start = 0usize;
        let mut missing: Vec<(usize, usize, u64)> = Vec::new();
        while start < p {
            let len = BLOCK_TOKENS.min(p - start);
            id = block_id(id, &tokens[start..start + len]);
            // Keep the chain complete from token 0: a block whose ring rows were recycled is
            // still stored (with its tokens and any full-attention rows), so `matched` can walk
            // across it to the newest snapshot.
            if !self.find(id, &tokens[start..start + len]) {
                missing.push((start, len, id));
            }
            start += len;
        }
        if missing.is_empty() {
            self.attach_snapshots(&kv.recurrent_snaps, &kv.cached, p, None)?;
            return Ok(0);
        }
        // One download per layer from the first new block; blocks already stored in between are
        // read and dropped, and rows a ring recycled come back absent.
        let from = missing[0].0;
        let mut all = KvRows::default();
        DeviceKv::new(SLOT, be, cfg, kv).get(SLOT, from..p, &mut all)?;
        for (start, len, id) in &missing {
            let mut rows = KvRows::default();
            rows.append(&all, start - from..start - from + len)?;
            rows.snap = None;
            let block = tokens[*start..*start + *len].to_vec();
            self.insert(*id, block, rows)?;
        }
        self.attach_snapshots(&kv.recurrent_snaps, &kv.cached, p, all.snap.take())?;
        self.captures += 1;
        Ok(p)
    }

    /// Attach the slot's recurrent-state snapshots to the blocks that end at their positions.
    /// Only snapshots whose whole chain is present can ever restore, and a block's state is the
    /// same whichever turn wrote it, so an already-attached snapshot is kept. `at_p` is a copy of
    /// the snapshot at `p` already taken, used instead of cloning it again.
    fn attach_snapshots(
        &mut self,
        snaps: &[RecurrentSnap],
        tokens: &[u32],
        p: usize,
        mut at_p: Option<Arc<RecurrentSnap>>,
    ) -> Result<()> {
        for snap in snaps {
            if snap.pos == 0 || snap.pos > p {
                continue;
            }
            let Some(id) = self.block_at(tokens, snap.pos) else {
                continue;
            };
            if !self.index[&id].snap {
                let shared = match at_p.take() {
                    Some(s) if s.pos == snap.pos => s,
                    other => {
                        at_p = other;
                        Arc::new(snap.clone())
                    }
                };
                self.attach(id, shared)?;
                tracing::debug!(pos = snap.pos, "kv store: attached a recurrent snapshot");
            }
        }
        Ok(())
    }

    /// The stored block ending at token `pos`, when the chain covers it exactly.
    fn block_at(&self, tokens: &[u32], pos: usize) -> Option<u64> {
        if pos == 0 || !pos.is_multiple_of(BLOCK_TOKENS) || pos > tokens.len() {
            return None;
        }
        let mut id = 0u64;
        let mut start = 0usize;
        while start < pos {
            let len = BLOCK_TOKENS.min(pos - start);
            id = block_id(id, &tokens[start..start + len]);
            start += len;
        }
        self.find(id, &tokens[pos - BLOCK_TOKENS..pos])
            .then_some(id)
    }
}

/// Strip credentials from a URL before it reaches a log line.
pub(crate) fn redact(url: &str) -> String {
    match url.split_once("://") {
        Some((scheme, rest)) => match rest.rsplit_once('@') {
            Some((_, host)) => format!("{scheme}://{host}"),
            None => url.to_string(),
        },
        None => url.to_string(),
    }
}

/// The store `kv.tiers` declares: one backend per entry, fastest first, folded so an eviction from
/// one entry demotes into the next and a read promotes back up. `None` when the cache is off. A
/// tier that cannot be opened costs only that tier: log and keep the rest.
pub(crate) fn prefix_store_for(kv: &crate::EngineConfig, key: u64) -> Option<Box<dyn KvStore>> {
    let tiers = &kv.kv.tiers;
    if tiers.is_empty() {
        return None;
    }
    let mut chain: Option<Box<dyn KvStore>> = None;
    for (i, tier) in tiers.iter().enumerate().rev() {
        let opened = match open_tier(tier, key) {
            Ok(opened) => opened,
            Err(e) => {
                tracing::warn!(tier = %tier.describe(), error = %e, "kv store: dropping a tier");
                continue;
            }
        };
        tracing::info!(tier = %tier.describe(), "kv store: tier {i} of {}", tiers.len());
        chain = match chain {
            Some(slow) => match opened {
                Opened::Budgeted(fast) => Some(Box::new(Tiered::new(fast, slow))),
                Opened::Plain(_) => {
                    tracing::warn!(
                        tier = %tier.describe(),
                        "kv store: a tier that cannot evict must be the last one; dropping it"
                    );
                    Some(slow)
                }
            },
            None => Some(match opened {
                Opened::Budgeted(store) => Box::new(Bounded::new(store)),
                Opened::Plain(store) => store,
            }),
        };
    }
    chain
}

/// A tier that can demote (it has a budget) or one that cannot (the server owns its eviction).
enum Opened {
    Budgeted(Box<dyn BudgetedStore>),
    Plain(Box<dyn KvStore>),
}

/// A redb tier's budget when its entry names none, which is what `--set kv.tiers=redb` produces.
const DEFAULT_TIER_BYTES: u64 = 1 << 30;

fn open_tier(tier: &infr_core::config::KvTier, key: u64) -> anyhow::Result<Opened> {
    match tier.backend.trim().to_ascii_lowercase().as_str() {
        "redb" => {
            if tier.url.is_some() {
                return Err(anyhow!(
                    "a redb tier keeps its blocks in `path`, and `url` belongs to a redis tier"
                ));
            }
            let size = match tier.size {
                None => DEFAULT_TIER_BYTES,
                Some(infr_core::SizeSpec::Percent(_)) => {
                    return Err(anyhow!("a tier `size` is bytes, not a share of something"))
                }
                Some(infr_core::SizeSpec::Bytes(0)) => {
                    return Err(anyhow!("a tier `size` has to be more than zero"))
                }
                Some(infr_core::SizeSpec::Bytes(bytes)) => bytes,
            };
            let store = match &tier.path {
                Some(path) => RedbKv::open_file(path, key, size as usize)?,
                None => RedbKv::open_memory(key, size as usize)?,
            };
            Ok(Opened::Budgeted(Box::new(store)))
        }
        "redis" | "valkey" => {
            if tier.path.is_some() {
                return Err(anyhow!(
                    "a redis tier is named by `url`, and `path` belongs to a redb tier"
                ));
            }
            if tier.size.is_some() {
                tracing::warn!(
                    tier = %tier.describe(),
                    "kv store: a redis tier is budgeted by the server, so `size` is ignored"
                );
            }
            let url = tier
                .url
                .as_deref()
                .ok_or_else(|| anyhow!("a redis tier needs a `url`"))?;
            RedisKv::connect(url, key)
                .map(|store| Opened::Plain(Box::new(store)))
                .map_err(|e| anyhow!("{}: {e}", redact(url)))
        }
        other => Err(anyhow!("unknown backend {other:?} (redb, redis)")),
    }
}

/// The device slot's extent id inside [`DeviceKv`] while the cache reads or writes it.
const SLOT: u64 = 0;

/// Pages `p` still has to allocate to hold rows `from..rows` that the pool does not have free.
/// A page it shares counts once, since the first write splits it.
fn page_deficit(p: &PagedSlot, from: usize, rows: usize) -> usize {
    let pool = p.map.pool();
    let rows = rows.min(pool.slot_rows());
    let first = from.min(rows) / PAGE_ROWS;
    let need = (first..rows.div_ceil(PAGE_ROWS))
        .filter(|&j| !p.map.is_private(j))
        .count();
    need.saturating_sub(pool.free_pages())
}

/// Write restored rows `first..n` into `kv`, splitting any page they share first.
fn upload_prefix(
    be: &dyn Backend,
    cfg: &Config,
    kv: &mut SeamKv,
    first: usize,
    n: usize,
    rows: KvRows,
) -> Result<()> {
    if let Some(p) = kv.paged.as_mut() {
        p.make_private(be, first, n)?;
    }
    DeviceKv::new(SLOT, be, cfg, kv).put(SLOT, first, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::kv_tier::{decode_rows, encode_rows, BudgetedStore, LayerRows};
    use crate::seam::rows::{read_rows, write_rows};
    use std::path::Path;

    type Cache = BlockCache<RedbKv>;

    /// A cache over redb's in-memory backend, unbounded.
    fn cache() -> Cache {
        Cache::new(RedbKv::open_memory(0, 1 << 20).unwrap(), true)
    }

    /// The first `n` rows of extent `id`, read back through the store's own `get`.
    fn held<S: KvStore>(store: &mut S, id: u64, n: usize) -> KvRows {
        let mut out = KvRows::default();
        assert!(store.get(id, 0..n, &mut out).unwrap(), "extent {id:x}");
        out
    }

    fn tokens(seed: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| seed * 1000 + i).collect()
    }

    /// Index the chain for `tokens` as if it had just been captured, each block counted as
    /// `bytes`. Returns the block ids.
    fn seed(s: &mut Cache, tokens: &[u32], bytes: usize) -> Vec<u64> {
        let mut id = 0u64;
        let mut n = 0usize;
        let mut ids = Vec::new();
        while n < tokens.len() {
            let len = BLOCK_TOKENS.min(tokens.len() - n);
            let block = &tokens[n..n + len];
            id = block_id(id, block);
            let rows = KvRows {
                rows: len,
                ..KvRows::default()
            };
            s.insert(id, block.to_vec(), rows).unwrap();
            s.index.get_mut(&id).unwrap().bytes = bytes;
            s.write_meta(id).unwrap();
            ids.push(id);
            n += len;
        }
        ids
    }

    fn entry<'a>(s: &'a Cache, tokens: &[u32]) -> &'a Entry {
        s.index
            .values()
            .find(|e| e.tokens == tokens)
            .expect("a block with these tokens")
    }

    #[test]
    fn matched_walks_the_chain_and_stops_at_a_gap() {
        let a = tokens(1, 300);
        let mut s = cache();
        seed(&mut s, &a, 1);
        assert_eq!(s.matched(&a).0, 300);
        let mut extended = a.clone();
        extended.extend([7, 8, 9]);
        // The short tail block only matches a prompt that ends there.
        assert_eq!(s.matched(&extended).0, 256);
        let mut diverged = a.clone();
        diverged[150] = 99;
        assert_eq!(s.matched(&diverged).0, BLOCK_TOKENS);
    }

    #[test]
    fn a_gone_extent_forgets_its_entry_and_stops_the_walk() {
        let a = tokens(1, 300);
        let mut s = cache();
        let ids = seed(&mut s, &a, 1);
        s.store.remove(ids[1]).unwrap();
        assert_eq!(s.matched(&a).0, BLOCK_TOKENS);
        assert!(!s.index.contains_key(&ids[1]));
    }

    #[test]
    fn shared_blocks_are_stored_once() {
        let a = tokens(1, 300);
        let mut b = a[..256].to_vec();
        b.extend(tokens(2, 200));
        let mut s = cache();
        seed(&mut s, &a, 1);
        assert_eq!(s.index.len(), 3);
        // The second prompt shares two blocks and adds two of its own.
        assert_eq!(s.matched(&b).0, 256);
        seed(&mut s, &b[256..], 1);
        assert_eq!(s.index.len(), 5);
        assert_eq!(entry(&s, &b[384..]).tokens.len(), 72);
    }

    #[test]
    fn block_at_names_only_a_complete_chain_position() {
        let a = tokens(1, 300);
        let mut s = cache();
        seed(&mut s, &a, 1);
        assert!(s.block_at(&a, 0).is_none());
        assert!(s.block_at(&a, 128).is_some());
        assert!(s.block_at(&a, 256).is_some());
        // The short tail block only answers for its own end.
        assert!(s.block_at(&a, 300).is_none());
        // A different token at the position breaks the chain hash.
        let mut b = a.clone();
        b[128] ^= 1;
        assert!(s.block_at(&b, 256).is_none());
    }

    fn snap(pos: usize) -> RecurrentSnap {
        RecurrentSnap {
            pos,
            k: vec![(0, vec![7u8; 64])],
            v: vec![(0, vec![9u8; 32])],
            bytes: 96,
        }
    }

    #[test]
    fn a_snapshot_rides_the_block_that_ends_at_it() {
        let a = tokens(1, 512);
        let mut s = cache();
        seed(&mut s, &a, 1);
        s.attach_snapshots(&[snap(512)], &a, 512, None).unwrap();
        let id = s.block_at(&a, 512).expect("block at 512");
        assert_eq!(s.index[&id].bytes, 96);
        assert!(s.index[&id].snap);
        let snap = held(&mut s.store, id, BLOCK_TOKENS).snap;
        assert_eq!(snap.expect("the snapshot in the block extent").pos, 512);
        // The stored metadata names the snapshot, so a reload rebuilds the index.
        let (_, snap_flag, bytes) = decode_meta(&s.store.meta(id).unwrap().unwrap()).unwrap();
        assert!(snap_flag);
        assert_eq!(bytes, 96);
    }

    #[test]
    fn formats_load_from_the_store() {
        let mut store = RedbKv::open_memory(0, 1 << 20).unwrap();
        let blob = encode_formats(DType::Q8_0, DType::F16).unwrap();
        store.set_meta(FORMATS_META_ID, &blob).unwrap();
        assert_eq!(decode_formats(&blob), Some((DType::Q8_0, DType::F16)));
        let cache = BlockCache::new(store, true);
        assert_eq!((cache.k_fmt, cache.v_fmt), (DType::Q8_0, DType::F16));
    }

    /// The chain id must not move with the toolchain: it is the key of every block in a lower
    /// tier, so a hash change would silently turn every stored prefix into a miss.
    #[test]
    fn block_ids_are_stable_across_releases() {
        assert_eq!(block_id(0, &[1, 2, 3]), 0x973d_5966_9a25_a835);
    }

    #[test]
    #[ignore = "needs a throwaway Redis/Valkey in INFR_TEST_REDIS_URL"]
    fn a_redis_store_round_trips_extents_and_metas() {
        let url = std::env::var("INFR_TEST_REDIS_URL").expect("set INFR_TEST_REDIS_URL");
        let key = 0x1234_5678_9abc_def0 ^ u64::from(std::process::id());
        let mut store = RedisKv::connect(&url, key).expect("connect");
        for id in store.extents().expect("list") {
            store.remove(id).expect("clear");
        }
        let a = KvRows::whole(3, vec![vec![1; 6]], vec![vec![4; 3]]);
        let mut b = KvRows::whole(6, vec![vec![6; 12]], vec![vec![8; 6]]);
        b.snap = Some(Arc::new(RecurrentSnap {
            pos: 6,
            k: vec![(0, vec![7; 8])],
            v: vec![(1, vec![9; 4])],
            bytes: 12,
        }));
        store.put(11, 0, a.clone()).unwrap();
        store.put(22, 0, b.clone()).unwrap();
        store.set_meta(11, b"three").unwrap();
        store.set_meta(22, b"six").unwrap();
        assert_eq!(store.extents().unwrap(), vec![11, 22]);
        assert_eq!(store.meta(11).unwrap().as_deref(), Some(&b"three"[..]));

        let mut got = KvRows::default();
        assert!(store.get(22, 0..6, &mut got).unwrap());
        assert_eq!(got.layers, b.layers);
        let snap = got.snap.expect("the snapshot");
        assert_eq!((snap.pos, snap.k[0].1.len()), (6, 8));

        store.remove(11).unwrap();
        assert_eq!(store.extents().unwrap(), vec![22]);
        assert!(!store.contains(11));
        let mut gone = KvRows::default();
        assert!(!store.get(11, 0..3, &mut gone).unwrap());
        store.remove(22).unwrap();
        assert!(store.extents().unwrap().is_empty());
    }

    #[test]
    fn the_rows_codec_keeps_partial_layers_and_the_snapshot() {
        let mut split = KvRows::whole(3, vec![vec![1; 6]], vec![vec![2; 3]]);
        split.layers.push(LayerRows {
            k: vec![3; 4],
            v: vec![4; 2],
            runs: vec![0..1, 2..3],
        });
        split.snap = Some(Arc::new(RecurrentSnap {
            pos: 3,
            k: vec![(1, vec![5; 3])],
            v: vec![(2, vec![6; 2])],
            bytes: 5,
        }));
        let blob = encode_rows(0, &split);
        let (first, got) = decode_rows(&blob).unwrap();
        assert_eq!(first, 0);
        assert_eq!(got.rows, 3);
        assert_eq!(got.layers[1].runs, vec![0..1, 2..3]);
        assert_eq!(got.layers[1].k, split.layers[1].k);
        let snap = got.snap.expect("snapshot");
        assert_eq!(
            (snap.pos, snap.k.clone(), snap.bytes),
            (3, vec![(1, vec![5; 3])], 5)
        );
    }

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("a temporary directory")
    }

    /// Three extents, each in its own `Durability::None` commit, then one flush on drop: the next
    /// open must see all three.
    #[test]
    fn three_extents_in_separate_commits_survive_a_reopen() {
        let dir = temp_dir();
        let path = dir.path().join("kv.redb");
        {
            let mut d = RedbKv::open_file(&path, 1, 1 << 20).unwrap();
            for id in 1..=3u64 {
                d.put(
                    id,
                    0,
                    KvRows::whole(1, vec![vec![id as u8; 4]], vec![vec![0; 2]]),
                )
                .unwrap();
                d.set_meta(id, b"m").unwrap();
            }
        }
        let mut d = RedbKv::open_file(&path, 1, 1 << 20).unwrap();
        let mut ids = d.extents().unwrap();
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![1, 2, 3],
            "every commit is in the file after the flush"
        );
    }

    #[test]
    fn a_disk_store_round_trips_and_survives_a_reopen() {
        let dir = temp_dir();
        let path = dir.path().join("kv.redb");
        let key = 0x55u64;
        {
            let mut d = RedbKv::open_file(&path, key, 1 << 20).unwrap();
            let a = KvRows::whole(2, vec![vec![1; 8]], vec![vec![2; 4]]);
            d.put(1, 0, a.clone()).unwrap();
            d.set_meta(1, b"one").unwrap();
            d.put(2, 0, KvRows::whole(3, vec![vec![3; 12]], vec![vec![4; 6]]))
                .unwrap();
            d.set_meta(2, b"two").unwrap();
            assert!(d.contains(1));
            let mut out = KvRows::default();
            assert!(d.get(1, 0..2, &mut out).unwrap());
            assert_eq!(out.layers, a.layers);
            assert_eq!(d.meta(1).unwrap().as_deref(), Some(&b"one"[..]));
            let mut ids = d.extents().unwrap();
            ids.sort_unstable();
            assert_eq!(ids, vec![1, 2]);
            d.remove(2).unwrap();
            assert!(!d.contains(2));
        }
        {
            let mut d = RedbKv::open_file(&path, key, 1 << 20).unwrap();
            assert!(d.contains(1));
            assert_eq!(d.meta(1).unwrap().as_deref(), Some(&b"one"[..]));
            let mut out = KvRows::default();
            assert!(d.get(1, 0..2, &mut out).unwrap());
            assert_eq!(out.layers[0].k, vec![1; 8]);
            assert_eq!(d.extents().unwrap(), vec![1]);
        }
        // Another model's key sees none of it.
        let mut other = RedbKv::open_file(&path, key + 1, 1 << 20).unwrap();
        assert!(other.extents().unwrap().is_empty());
    }

    #[test]
    fn a_journal_file_is_not_mistaken_for_a_database() {
        let dir = temp_dir();
        let path = dir.path().join("kv.journal");
        std::fs::write(&path, b"INFRKV02 not a redb database").unwrap();
        assert!(RedbKv::open_file(&path, 1, 1 << 20).is_err());
    }

    #[test]
    fn a_disk_store_takes_its_lru_in_order() {
        let dir = temp_dir();
        let path = dir.path().join("kv.redb");
        let mut d = RedbKv::open_file(&path, 1, 1 << 20).unwrap();
        d.put(1, 0, KvRows::whole(1, vec![vec![1]], vec![vec![2]]))
            .unwrap();
        d.put(2, 0, KvRows::whole(1, vec![vec![3]], vec![vec![4]]))
            .unwrap();
        let mut out = KvRows::default();
        d.get(1, 0..1, &mut out).unwrap();
        let taken = d.take_lru().expect("an extent");
        assert_eq!((taken.id, taken.first, taken.rows.rows), (2, 0, 1));
        assert!(!d.contains(2));
        assert!(d.contains(1));
    }

    #[test]
    fn tiered_demotes_on_overflow_and_promotes_on_read() {
        let mut t = Tiered::new(
            RedbKv::open_memory(0, 8).unwrap(),
            RedbKv::open_memory(1, 0).unwrap(),
        );
        let a = KvRows::whole(1, vec![vec![1; 4]], vec![vec![2; 2]]);
        let b = KvRows::whole(1, vec![vec![3; 4]], vec![vec![4; 2]]);
        t.put(1, 0, a.clone()).unwrap();
        t.put(2, 0, b.clone()).unwrap();
        assert!(!t.fast().contains(1), "the older extent left RAM");
        assert!(t.slow().contains(1), "and landed in the slow tier");
        assert!(t.fast().contains(2));
        assert_eq!(held(t.fast_mut(), 2, 1).layers, b.layers);
        assert_eq!(held(t.slow_mut(), 1, 1).layers, a.layers);

        let mut got = KvRows::default();
        assert!(t.get(1, 0..1, &mut got).unwrap());
        assert_eq!(got.layers, a.layers);
        assert!(t.fast().contains(1), "a read promotes the extent");
        assert!(t.slow().contains(1), "the slow copy stays");
        assert_eq!(held(t.fast_mut(), 1, 1).layers, a.layers);
        assert_eq!(held(t.slow_mut(), 1, 1).layers, a.layers);
    }

    #[test]
    fn tiered_keeps_the_only_extent_even_over_budget() {
        let mut t = Tiered::new(
            RedbKv::open_memory(0, 4).unwrap(),
            RedbKv::open_memory(1, 0).unwrap(),
        );
        let a = KvRows::whole(1, vec![vec![1; 4]], vec![vec![2; 2]]);
        t.put(1, 0, a.clone()).unwrap();
        assert!(t.fast().contains(1));
        assert!(!t.slow().contains(1));
    }

    /// A promotion must not leave the fast tier over budget: the read that just happened stays,
    /// and whatever became least recently used goes down.
    #[test]
    fn a_promotion_settles_the_fast_tier() {
        let mut t = Tiered::new(
            RedbKv::open_memory(0, 8).unwrap(),
            RedbKv::open_memory(1, 0).unwrap(),
        );
        t.put(1, 0, KvRows::whole(1, vec![vec![1; 4]], vec![vec![2; 2]]))
            .unwrap();
        t.put(2, 0, KvRows::whole(1, vec![vec![3; 4]], vec![vec![4; 2]]))
            .unwrap();
        assert!(t.slow().contains(1) && t.fast().contains(2));

        let mut got = KvRows::default();
        assert!(t.get(1, 0..1, &mut got).unwrap());
        assert!(t.fast().contains(1), "the read stays in RAM");
        assert!(!t.fast().contains(2), "the other block went down");
        assert!(t.slow().contains(2));
        assert!(t.fast().used() <= t.fast().capacity());
    }

    /// The index entry travels with the rows: a promoted block carries the meta that names its
    /// tokens up with it, so a start that loads the fast tier alone still finds the block.
    #[test]
    fn a_promotion_carries_the_index_entry_up() {
        let mut t = Tiered::new(
            RedbKv::open_memory(0, 8).unwrap(),
            RedbKv::open_memory(1, 0).unwrap(),
        );
        let rows = KvRows::whole(1, vec![vec![1; 4]], vec![vec![2; 2]]);
        t.put(1, 0, rows).unwrap();
        t.fast_mut().set_meta(1, b"entry").unwrap();
        t.slow_mut().set_meta(1, b"entry").unwrap();
        t.put(2, 0, KvRows::whole(1, vec![vec![3; 4]], vec![vec![4; 2]]))
            .unwrap();
        assert!(!t.fast().contains(1), "1 was demoted by the second write");

        let mut got = KvRows::default();
        assert!(t.get(1, 0..1, &mut got).unwrap());
        assert_eq!(
            t.fast_mut().meta(1).unwrap().as_deref(),
            Some(&b"entry"[..])
        );
    }

    /// A redb tier gets 1 GiB when its entry names no size, which is what `--set kv.tiers=redb`
    /// builds; a field that belongs to the other backend is refused instead of ignored.
    #[test]
    fn a_tier_is_checked_against_its_backend() {
        use infr_core::SizeSpec;
        let tier = |backend: &str| infr_core::config::KvTier {
            backend: backend.to_string(),
            ..Default::default()
        };
        let Ok(Opened::Budgeted(redb)) = open_tier(&tier("redb"), 7) else {
            panic!("a redb tier with no size opens")
        };
        assert_eq!(redb.capacity(), DEFAULT_TIER_BYTES as usize);

        let wrong_url = infr_core::config::KvTier {
            url: Some("redis://127.0.0.1:1/0".into()),
            ..tier("redb")
        };
        assert!(open_tier(&wrong_url, 7).is_err(), "redb has no url");

        let wrong_path = infr_core::config::KvTier {
            path: Some(std::path::PathBuf::from("/tmp/kv.redb")),
            ..tier("redis")
        };
        assert!(open_tier(&wrong_path, 7).is_err(), "redis has no path");

        for size in [SizeSpec::Bytes(0), SizeSpec::Percent(50.0)] {
            let bad = infr_core::config::KvTier {
                size: Some(size),
                ..tier("redb")
            };
            assert!(open_tier(&bad, 7).is_err(), "{size:?} is not a budget");
        }
        assert!(open_tier(&tier("lmcache"), 7).is_err(), "unknown backend");
    }

    /// A block rewritten while an older copy sits below it must not leave that copy behind: the
    /// next start would load the stale rows, snapshot flag and all.
    #[test]
    fn a_rewritten_extent_drops_the_stale_copy_below_it() {
        let mut t = Tiered::new(
            RedbKv::open_memory(0, 8).unwrap(),
            RedbKv::open_memory(1, 0).unwrap(),
        );
        t.put(1, 0, KvRows::whole(1, vec![vec![1; 4]], vec![vec![2; 2]]))
            .unwrap();
        t.put(2, 0, KvRows::whole(1, vec![vec![3; 4]], vec![vec![4; 2]]))
            .unwrap();
        assert!(t.slow().contains(1));
        t.set_meta(1, b"old").unwrap();

        let newer = KvRows::whole(1, vec![vec![9; 4]], vec![vec![9; 2]]);
        t.put(1, 0, newer.clone()).unwrap();
        assert!(!t.slow().contains(1), "the stale copy is gone");
        assert!(t.fast().contains(1));
        let mut got = KvRows::default();
        assert!(t.get(1, 0..1, &mut got).unwrap());
        assert_eq!(got.layers, newer.layers);
    }

    /// A budgeted store with nothing below it drops its own LRU instead of growing past the cap.
    #[test]
    fn a_bounded_store_drops_its_own_lru() {
        let mut b = Bounded::new(RedbKv::open_memory(0, 8).unwrap());
        b.put(1, 0, KvRows::whole(1, vec![vec![1; 4]], vec![vec![2; 2]]))
            .unwrap();
        b.put(2, 0, KvRows::whole(1, vec![vec![3; 4]], vec![vec![4; 2]]))
            .unwrap();
        assert!(!b.contains(1), "the older extent was dropped");
        assert!(b.contains(2));
    }

    const ROW: usize = 32;

    /// Two attention layers of 32-element f16 rows; with `window`, layer 0 is a sliding window.
    fn slot_cfg(window: usize) -> Config {
        Config {
            n_layer: 2,
            n_head: 4,
            n_kv: 2,
            n_embd: 64,
            head_dim: 16,
            rope_dim: 16,
            head_dim_swa: 16,
            n_kv_swa: 2,
            swa_window: window,
            swa_pattern: if window > 0 { 2 } else { 0 },
            ..Default::default()
        }
    }

    /// The bytes of logical row `r` of layer `l`'s side `side` in the test slots.
    fn row_data(l: usize, side: u16, r: usize) -> Vec<u8> {
        let v = (r as u16) ^ ((l as u16) << 12) ^ (side << 14);
        (0..ROW).flat_map(|_| v.to_le_bytes()).collect()
    }

    /// A slot holding `n` tokens, every row written through its own layout.
    fn filled_slot(be: &dyn Backend, cfg: &Config, ring: bool, n: usize) -> SeamKv {
        let mut kv = SeamKv::for_test(be, cfg, 1024, ring, 56, DType::F16).unwrap();
        for l in 0..cfg.n_layer {
            let layout = kv.row_layout(be, cfg, l);
            for (side, buf) in [(0, &kv.kbufs[l]), (1, &kv.vbufs[l])] {
                let data: Vec<u8> = (0..n).flat_map(|r| row_data(l, side, r)).collect();
                write_rows(be, buf.as_ref(), DType::F16, ROW, &*layout, 0..n, &data).unwrap();
            }
        }
        kv.cached = tokens(3, n);
        kv.written_end = n;
        kv
    }

    fn read_side(
        be: &dyn Backend,
        cfg: &Config,
        kv: &SeamKv,
        l: usize,
        side: u16,
        rows: std::ops::Range<usize>,
    ) -> Vec<u8> {
        let layout = kv.row_layout(be, cfg, l);
        let buf = if side == 0 {
            &kv.kbufs[l]
        } else {
            &kv.vbufs[l]
        };
        let mut out = vec![0u8; rows.len() * ROW * 2];
        read_rows(be, buf.as_ref(), DType::F16, ROW, &*layout, rows, &mut out).unwrap();
        out
    }

    /// A capture keeps the reply's rows too; a recurrent model can only resume at a snapshot,
    /// which prefill takes at prompt boundaries, so its reply rows would never be read back.
    #[test]
    fn capture_upto_follows_what_the_slot_wrote() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(0);
        let mut kv = SeamKv::for_test(&be, &cfg, 1024, false, 56, DType::F16).unwrap();
        kv.cached = tokens(1, 400);
        assert_eq!(capture_upto(&cfg, &kv, 256), 400);

        let rec = Config {
            lfm3: true,
            shortconv_layers: vec![true, false],
            shortconv_l_cache: 4,
            ..slot_cfg(0)
        };
        assert!(rec.has_recurrent_state());
        assert_eq!(capture_upto(&rec, &kv, 256), 256);
    }

    #[test]
    fn a_linear_slot_round_trips_through_the_cache() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(0);
        let src = filled_slot(&be, &cfg, false, 300);
        let mut s = cache();
        assert_eq!(s.capture(&be, &cfg, &src, 300).unwrap(), 300);
        assert_eq!(s.index.len(), 3);
        // Capturing the same prefix again stores nothing new.
        assert_eq!(s.capture(&be, &cfg, &src, 300).unwrap(), 0);
        assert_eq!(s.captures, 1);

        let mut dst = SeamKv::for_test(&be, &cfg, 1024, false, 56, DType::F16).unwrap();
        assert_eq!(s.restore(&be, &cfg, &mut dst, &src.cached).unwrap(), 300);
        assert_eq!(dst.cached, src.cached);
        assert_eq!((s.restores, s.restored_tokens), (1, 300));
        for l in 0..2 {
            for side in 0..2 {
                let want: Vec<u8> = (0..300).flat_map(|r| row_data(l, side, r)).collect();
                assert_eq!(
                    read_side(&be, &cfg, &dst, l, side, 0..300),
                    want,
                    "layer {l} side {side}"
                );
            }
        }
    }

    /// A wrapped ring keeps only its window: the block that straddles the window's start stores
    /// the rows of it that are still resident, and the restore writes the window back in place.
    #[test]
    fn a_wrapped_ring_round_trips_its_window() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(200);
        let n = 700;
        let src = filled_slot(&be, &cfg, true, n);
        let ring = src.window_ring_rows(&be, &cfg).expect("a ring");
        assert_eq!(ring, 256);
        let mut s = cache();
        assert_eq!(s.capture(&be, &cfg, &src, n).unwrap(), n);
        let ids = s.matched(&src.cached).1;
        assert_eq!(ids.len(), 6);
        // The window starts at row 444: blocks below it are empty, the one straddling it keeps
        // the rows from 444 on, and the rest are whole.
        for (j, id) in ids.iter().enumerate() {
            let len = BLOCK_TOKENS.min(n - j * BLOCK_TOKENS);
            let rows = held(&mut s.store, *id, len);
            let runs = &rows.layers[0].runs;
            let run = (runs.len() == 1).then(|| runs[0].clone());
            match j {
                0..=2 => assert!(runs.is_empty(), "block {j}"),
                3 => assert_eq!(run, Some(60..128), "block {j}"),
                4 => assert_eq!(run, Some(0..BLOCK_TOKENS), "block {j}"),
                _ => assert_eq!(run, Some(0..60), "block {j}"),
            }
            assert!(!rows.layers[1].k.is_empty());
        }

        let mut dst = SeamKv::for_test(&be, &cfg, 1024, true, 56, DType::F16).unwrap();
        assert_eq!(s.restore(&be, &cfg, &mut dst, &src.cached).unwrap(), n);
        for side in 0..2 {
            for l in 0..2 {
                let from = dst.row_layout(&be, &cfg, l).oldest_resident(n);
                let want: Vec<u8> = (from..n).flat_map(|r| row_data(l, side, r)).collect();
                assert_eq!(
                    read_side(&be, &cfg, &dst, l, side, from..n),
                    want,
                    "layer {l} side {side} from row {from}"
                );
            }
        }
    }

    /// A paged slot of `n` written rows over `pool`, its pages reserved through `n`.
    fn paged_slot(
        be: &dyn Backend,
        cfg: &Config,
        pool: &Arc<crate::seam::KvPool>,
        n: usize,
    ) -> SeamKv {
        let mut kv = SeamKv::for_test(be, cfg, 1024, false, 56, DType::F16).unwrap();
        kv.attach_paged(be, cfg, pool).unwrap();
        kv.paged.as_mut().unwrap().reserve_rows(be, n).unwrap();
        for l in 0..cfg.n_layer {
            let layout = kv.row_layout(be, cfg, l);
            for (side, buf) in [(0, &kv.kbufs[l]), (1, &kv.vbufs[l])] {
                let data: Vec<u8> = (0..n).flat_map(|r| row_data(l, side, r)).collect();
                write_rows(be, buf.as_ref(), DType::F16, ROW, &*layout, 0..n, &data).unwrap();
            }
        }
        kv.cached = tokens(3, n);
        kv.written_end = n;
        kv
    }

    fn assert_rows(be: &dyn Backend, cfg: &Config, kv: &SeamKv, rows: std::ops::Range<usize>) {
        for l in 0..cfg.n_layer {
            for side in 0..2 {
                let want: Vec<u8> = rows.clone().flat_map(|r| row_data(l, side, r)).collect();
                let got = read_side(be, cfg, kv, l, side, rows.clone());
                assert_eq!(got, want, "layer {l} side {side}");
            }
        }
    }

    /// A parked paged slot keeps its whole pages; resuming adopts them by reference, so only the
    /// partial page past them is read from the store, and the adopter's write into a shared page
    /// splits it instead of touching the resident's copy.
    #[test]
    fn a_parked_paged_slot_resumes_from_its_pages() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(0);
        let pool = crate::seam::KvPool::new(&be, &cfg, 1024, DType::F16, DType::F16, 5).unwrap();
        let n = 600;
        let mut src = paged_slot(&be, &cfg, &pool, n);
        let mut s = cache();
        assert_eq!(s.capture(&be, &cfg, &src, n).unwrap(), n);
        assert!(
            s.retain(&cfg, &mut src),
            "a paged slot with a whole page is kept"
        );
        src.reset();
        assert_eq!(
            pool.used_pages(),
            1,
            "the partial page went back, the whole one stayed"
        );
        let page = s.resident[0].map.page(0);

        let mut dst = SeamKv::for_test(&be, &cfg, 1024, false, 56, DType::F16).unwrap();
        dst.attach_paged(&be, &cfg, &pool).unwrap();
        // The short tail block only matches at the end of a prompt: a resend of the same tokens.
        let prompt = tokens(3, n);
        assert_eq!(s.restore(&be, &cfg, &mut dst, &prompt).unwrap(), n);
        let p = dst.paged.as_ref().unwrap();
        assert_eq!(p.map.page(0), page, "the first page is the resident's");
        assert!(!p.map.is_private(0));
        assert_eq!((s.resident_hits, s.resident_tokens), (1, 512));
        assert_eq!(
            (s.restores, s.restored_tokens),
            (1, 88),
            "only the partial page came from the store"
        );
        assert_eq!(dst.cached, prompt[..n]);
        assert_rows(&be, &cfg, &dst, 0..n);

        dst.paged
            .as_mut()
            .unwrap()
            .make_private(&be, 500, 520)
            .unwrap();
        assert_ne!(dst.paged.as_ref().unwrap().map.page(0), page);
        assert_eq!(s.resident[0].map.page(0), page);
        assert_rows(&be, &cfg, &dst, 0..n);
    }

    /// Without a store, a resident is still adopted; spilling it forgets its blocks and gives
    /// its pages back.
    #[test]
    fn residents_work_without_a_store_and_spill_their_pages() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(0);
        let pool = crate::seam::KvPool::new(&be, &cfg, 1024, DType::F16, DType::F16, 5).unwrap();
        let mut s = Cache::new(RedbKv::open_memory(0, 1 << 20).unwrap(), false);
        let mut src = paged_slot(&be, &cfg, &pool, 1100);
        assert_eq!(s.capture(&be, &cfg, &src, 1100).unwrap(), 0);
        assert!(s.retain(&cfg, &mut src));
        src.reset();
        assert_eq!(pool.used_pages(), 2);
        assert_eq!(s.index.len(), 8, "two pages of four blocks each");

        let mut dst = SeamKv::for_test(&be, &cfg, 1024, false, 56, DType::F16).unwrap();
        dst.attach_paged(&be, &cfg, &pool).unwrap();
        let prompt = tokens(3, 1100);
        assert_eq!(s.restore(&be, &cfg, &mut dst, &prompt).unwrap(), 1024);
        assert_eq!((s.resident_hits, s.restores), (1, 0));
        assert_rows(&be, &cfg, &dst, 0..1024);
        dst.reset();

        assert_eq!(s.spill(1), 1);
        assert_eq!(s.resident_spills, 1);
        assert!(
            s.index.is_empty(),
            "blocks only the resident held are forgotten"
        );
        assert_eq!(pool.used_pages(), 0);
        assert_eq!(s.restore(&be, &cfg, &mut dst, &prompt).unwrap(), 0);
    }

    /// Free pages that cover the run's writes keep the residents: only a real shortfall spills.
    #[test]
    fn making_room_keeps_residents_the_free_pages_cover() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(0);
        let pool = crate::seam::KvPool::new(&be, &cfg, 2048, DType::F16, DType::F16, 8).unwrap();
        let mut s = cache();
        let mut src = paged_slot(&be, &cfg, &pool, 600);
        s.capture(&be, &cfg, &src, 600).unwrap();
        assert!(s.retain(&cfg, &mut src), "a whole page is kept");
        src.reset();

        let mut dst = SeamKv::for_test(&be, &cfg, 2048, false, 56, DType::F16).unwrap();
        dst.attach_paged(&be, &cfg, &pool).unwrap();
        let prompt = tokens(3, 600);
        assert_eq!(s.restore(&be, &cfg, &mut dst, &prompt).unwrap(), 600);
        assert!(pool.free_pages() >= 3, "room for the tail");

        assert_eq!(s.make_room(&dst, 1600), 0, "free pages cover the writes");
        assert_eq!(s.resident.len(), 1, "the resident survived");
        assert_eq!(s.resident_spills, 0);

        assert_eq!(s.make_room(&dst, 1024), 0, "the tail page is already held");
        assert_eq!(s.resident.len(), 1);
    }

    /// A restore whose blocks cannot fill the destination's window re-prefills instead of leaving
    /// rows attention will read unwritten.
    #[test]
    fn a_restore_into_a_wider_ring_refuses() {
        let be = infr_cpu::CpuBackend::new();
        let src_cfg = slot_cfg(200);
        let n = 700;
        let src = filled_slot(&be, &src_cfg, true, n);
        let mut s = cache();
        s.capture(&be, &src_cfg, &src, n).unwrap();

        let dst_cfg = slot_cfg(400);
        let mut dst = SeamKv::for_test(&be, &dst_cfg, 1024, true, 56, DType::F16).unwrap();
        assert_eq!(dst.window_ring_rows(&be, &dst_cfg), Some(512));
        assert_eq!(s.restore(&be, &dst_cfg, &mut dst, &src.cached).unwrap(), 0);
        assert!(dst.cached.is_empty());
    }

    /// Decode can write rows past the tokens it accepted, so the ring recycles from the row count
    /// actually written, not from the cached prefix.
    #[test]
    fn a_capture_skips_rows_a_chain_wrote_past_the_prompt() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(200);
        let mut src = filled_slot(&be, &cfg, true, 600);
        src.written_end = 800;
        let mut s = cache();
        assert_eq!(s.capture(&be, &cfg, &src, 600).unwrap(), 600);
        let ids = s.matched(&src.cached).1;
        assert_eq!(ids.len(), 5);
        // Rows below 544 are gone: the block over 512..640 keeps only 544..600.
        let rows = held(&mut s.store, ids[4], 88);
        assert_eq!(rows.layers[0].runs, vec![32..88]);
    }

    /// One block of the two-layer test slot, in bytes.
    fn block_bytes(cfg: &Config) -> usize {
        cfg.n_layer * BLOCK_TOKENS * ROW * 2 * 2
    }

    type RedbCache = BlockCache<Tiered<RedbKv, RedbKv>>;

    fn redb_cache(path: &Path, cap: usize, key: u64) -> RedbCache {
        let local = RedbKv::open_file(path, key, 1 << 20).unwrap();
        BlockCache::new(
            Tiered::new(RedbKv::open_memory(0, cap).unwrap(), local),
            true,
        )
    }

    fn assert_slot_rows(be: &dyn Backend, cfg: &Config, kv: &SeamKv, n: usize) {
        for l in 0..cfg.n_layer {
            for side in 0..2 {
                let want: Vec<u8> = (0..n).flat_map(|r| row_data(l, side, r)).collect();
                assert_eq!(
                    read_side(be, cfg, kv, l, side, 0..n),
                    want,
                    "layer {l} side {side}"
                );
            }
        }
    }

    #[test]
    fn a_block_evicted_from_ram_restores_from_disk_across_a_restart() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(0);
        let dir = temp_dir();
        let path = dir.path().join("kv.redb");
        let key = 0x99u64;
        let src = filled_slot(&be, &cfg, false, 300);
        let cap = block_bytes(&cfg);
        {
            let mut s = redb_cache(&path, cap, key);
            assert_eq!(s.capture(&be, &cfg, &src, 300).unwrap(), 300);
            let ids = s.matched(&src.cached).1;
            assert_eq!(ids.len(), 3);
            assert!(
                !s.store.fast().contains(ids[0]),
                "RAM dropped the first block"
            );
            assert!(
                s.store.slow().contains(ids[0]),
                "disk holds the first block"
            );
            assert!(s.store.fast().contains(ids[2]), "RAM kept the tail");

            let mut dst = SeamKv::for_test(&be, &cfg, 1024, false, 56, DType::F16).unwrap();
            assert_eq!(s.restore(&be, &cfg, &mut dst, &src.cached).unwrap(), 300);
            assert_slot_rows(&be, &cfg, &dst, 300);
        }
        {
            // The restart path: the index comes back from extents and metas alone, and the
            // restore above moved every block through disk, so all three are there.
            let mut s = redb_cache(&path, cap, key);
            assert_eq!(s.index.len(), 3);
            assert_eq!(
                s.store.slow().block_reads(),
                0,
                "index loaded without block reads"
            );
            assert_eq!((s.k_fmt, s.v_fmt), (DType::F16, DType::F16));
            let mut dst = SeamKv::for_test(&be, &cfg, 1024, false, 56, DType::F16).unwrap();
            assert_eq!(s.restore(&be, &cfg, &mut dst, &src.cached).unwrap(), 300);
            assert_slot_rows(&be, &cfg, &dst, 300);
        }
    }

    /// A damaged extent is a cache miss, never a failed request: the block is dropped and the
    /// prompt re-prefills.
    #[test]
    fn a_damaged_disk_extent_is_a_miss() {
        use crate::seam::kv_tier::REDB_EXTENTS;

        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(0);
        let dir = temp_dir();
        let path = dir.path().join("kv.redb");
        let key = 0x77u64;
        let src = filled_slot(&be, &cfg, false, 300);
        let cap = block_bytes(&cfg);
        let first = {
            let mut s = redb_cache(&path, cap, key);
            s.capture(&be, &cfg, &src, 300).unwrap();
            let ids = s.matched(&src.cached).1;
            assert!(
                !s.store.fast().contains(ids[0]),
                "the oldest block went to disk"
            );
            ids[0]
        };
        {
            let db = redb::Database::create(&path).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let mut table = txn.open_table(REDB_EXTENTS).unwrap();
                table.insert((key, first), b"not rows".as_slice()).unwrap();
            }
            txn.commit().unwrap();
        }

        let mut s = redb_cache(&path, cap, key);
        assert!(s.index.contains_key(&first), "the index still names it");
        let mut dst = SeamKv::for_test(&be, &cfg, 1024, false, 56, DType::F16).unwrap();
        assert_eq!(s.restore(&be, &cfg, &mut dst, &src.cached).unwrap(), 0);
        assert!(dst.cached.is_empty());
        assert!(!s.store.contains(first), "the damaged extent was dropped");
        assert!(!s.index.contains_key(&first));
    }

    #[test]
    fn a_ram_cache_with_a_tiny_budget_keeps_and_misses_the_right_blocks() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(0);
        let a = filled_slot(&be, &cfg, false, 300);
        let mut b = filled_slot(&be, &cfg, false, 300);
        b.cached = tokens(4, 300);
        let cap = block_bytes(&cfg);
        let mut s: BlockCache<Tiered<RedbKv, RedbKv>> = BlockCache::new(
            Tiered::new(
                RedbKv::open_memory(0, cap).unwrap(),
                RedbKv::open_memory(1, 0).unwrap(),
            ),
            true,
        );
        assert_eq!(s.capture(&be, &cfg, &a, 300).unwrap(), 300);
        assert_eq!(s.capture(&be, &cfg, &b, 300).unwrap(), 300);
        let a_ids = s.matched(&a.cached).1;
        assert_eq!(a_ids.len(), 3);
        assert!(
            !s.store.fast().contains(a_ids[0]),
            "the RAM tier dropped the first block"
        );
        assert!(s.store.slow().contains(a_ids[0]));

        // What the tiers still hold restores, rows and all.
        let mut dst = SeamKv::for_test(&be, &cfg, 1024, false, 56, DType::F16).unwrap();
        assert_eq!(s.restore(&be, &cfg, &mut dst, &b.cached).unwrap(), 300);
        assert_slot_rows(&be, &cfg, &dst, 300);

        // A block gone from both tiers is a miss, not a half restore.
        for id in a_ids {
            s.store.remove(id).unwrap();
        }
        let mut dst = SeamKv::for_test(&be, &cfg, 1024, false, 56, DType::F16).unwrap();
        assert_eq!(s.restore(&be, &cfg, &mut dst, &a.cached).unwrap(), 0);
        assert!(dst.cached.is_empty());
    }
}
