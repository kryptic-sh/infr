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
use anyhow::Result;
use infr_core::backend::Backend;
use infr_core::DType;
use std::collections::HashMap;
use std::sync::Arc;

use super::{
    kv_tier::{
        decode_meta, encode_meta, Bounded, DeviceKv, DiskKv, Fnv, KvRows, KvStore, RedisKv, Tiered,
        FORMATS_META_ID,
    },
    weights::{RecurrentSnap, SeamKv},
};

/// Tokens per stored block. Smaller blocks share more (a 600-token system prompt with two
/// different tails stores each half once) at the price of one download per new block.
pub(crate) const BLOCK_TOKENS: usize = 128;

/// Below this many shared tokens the upload costs more than re-prefilling them.
pub(crate) const MIN_PREFIX_TOKENS: usize = 256;

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

/// Identity of the model a lower tier's blocks belong to: the geometry that shapes every row, plus
/// the context and vocabulary. Blocks under another key are never loaded.
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
    /// Whether `kv.store_mb` configured a cache at all.
    enabled: bool,
    /// The KV layouts the blocks were captured with; a dtype change invalidates every block.
    k_fmt: DType,
    v_fmt: DType,
    pub(crate) restores: u64,
    pub(crate) restored_tokens: u64,
    pub(crate) captures: u64,
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
            restores: 0,
            restored_tokens: 0,
            captures: 0,
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
        if self.index.remove(&id).is_none() {
            return;
        }
        if let Err(e) = self.store.remove(id) {
            tracing::warn!(error = %e, "kv store: dropping a block's extent failed");
        }
    }

    fn clear(&mut self) {
        let ids: Vec<u64> = self.index.keys().copied().collect();
        for id in ids {
            self.drop_entry(id);
        }
    }

    fn find(&self, id: u64, tokens: &[u32]) -> bool {
        self.index.get(&id).is_some_and(|e| e.tokens == tokens)
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
            if !self.store.contains(id) {
                self.index.remove(&id);
                break;
            }
            let Some(entry) = self.index.get_mut(&id) else {
                break;
            };
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
        if !self.enabled() {
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
        let mut rows = KvRows::default();
        for id in &keep {
            let len = self.index[id].tokens.len();
            match self.store.get(*id, 0..len, &mut rows) {
                Ok(true) => {}
                Ok(false) => return Ok(0),
                Err(e) => {
                    // A damaged extent is a cache miss, never a failed request.
                    tracing::warn!(id, error = %e, "kv store: dropping an unreadable block");
                    self.drop_entry(*id);
                    return Ok(0);
                }
            }
        }
        if let Some(p) = kv.paged.as_mut() {
            p.make_private(be, 0, n)?;
        }
        // Every row attention will read must be there: a window layer keeps only the newest rows
        // it has, and a restore that cannot fill that window must re-prefill instead.
        for l in 0..cfg.n_layer {
            if cfg.is_recurrent_layer(l) {
                continue;
            }
            let need = kv.row_layout(be, cfg, l).oldest_resident(n)..n;
            if !need.is_empty() && !rows.layers.get(l).is_some_and(|r| r.covers(need.clone())) {
                tracing::debug!(layer = l, need = ?need, "kv store: the stored blocks miss the window");
                return Ok(0);
            }
        }
        DeviceKv::new(SLOT, be, cfg, kv).put(SLOT, 0, rows)?;
        kv.written_end = n;
        kv.cached = prompt[..n].to_vec();
        self.restores += 1;
        self.restored_tokens += n as u64;
        Ok(n)
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

/// The tier below RAM that `kv.store_url` / `kv.store_file` select, nested disk-then-Redis when
/// both are set. A missing or broken tier only costs persistence: log and keep the RAM tier.
pub(crate) fn lower_tier_for(kv: &crate::EngineConfig, key: u64) -> Option<Box<dyn KvStore>> {
    let redis = kv
        .kv
        .store_url
        .as_deref()
        .map(str::trim)
        .filter(|u| !u.is_empty())
        .and_then(|url| match RedisKv::connect(url, key) {
            Ok(r) => {
                tracing::info!(endpoint = %r.describe(), "kv store: sharing the lower tier over Redis");
                Some(r)
            }
            Err(e) => {
                tracing::warn!(
                    endpoint = %redact(url),
                    error = %e,
                    "kv store: cannot reach the Redis tier"
                );
                None
            }
        });
    let disk = kv.kv.store_file.as_ref().and_then(|path| {
        let cap_mb = kv
            .kv
            .store_disk_mb
            .or_else(|| kv.kv.store_mb.map(|mb| mb.saturating_mul(4)))
            .unwrap_or(0);
        let cap = (cap_mb as usize).saturating_mul(1024 * 1024);
        match DiskKv::open(path, key, cap) {
            Ok(d) => Some(d),
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "kv store: cannot open the disk tier; keeping the RAM tier only"
                );
                None
            }
        }
    });
    match (disk, redis) {
        (Some(disk), Some(redis)) => Some(Box::new(Tiered::new(disk, redis))),
        (Some(disk), None) => Some(Box::new(Bounded::new(disk))),
        (None, Some(redis)) => Some(Box::new(redis)),
        (None, None) => None,
    }
}

/// The device slot's extent id inside [`DeviceKv`] while the cache reads or writes it.
const SLOT: u64 = 0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::kv_tier::RamKv;
    use crate::seam::kv_tier::{decode_rows, encode_rows, BudgetedStore, LayerRows};
    use crate::seam::rows::{read_rows, write_rows};
    use std::path::Path;

    type Cache = BlockCache<RamKv>;

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
        let mut s = Cache::new(RamKv::default(), true);
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
        let mut s = Cache::new(RamKv::default(), true);
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
        let mut s = Cache::new(RamKv::default(), true);
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
        let mut s = Cache::new(RamKv::default(), true);
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
        let mut s = Cache::new(RamKv::default(), true);
        seed(&mut s, &a, 1);
        s.attach_snapshots(&[snap(512)], &a, 512, None).unwrap();
        let id = s.block_at(&a, 512).expect("block at 512");
        assert_eq!(s.index[&id].bytes, 96);
        assert!(s.index[&id].snap);
        let held = s.store.host(id).and_then(|r| r.snap.clone());
        assert_eq!(held.expect("the snapshot in the block extent").pos, 512);
        // The stored metadata names the snapshot, so a reload rebuilds the index.
        let (_, snap_flag, bytes) = decode_meta(&s.store.meta(id).unwrap().unwrap()).unwrap();
        assert!(snap_flag);
        assert_eq!(bytes, 96);
    }

    #[test]
    fn formats_load_from_the_store() {
        let mut store = RamKv::default();
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

    #[test]
    fn a_disk_store_round_trips_and_survives_a_reopen() {
        let dir = temp_dir();
        let path = dir.path().join("kv.redb");
        let key = 0x55u64;
        {
            let mut d = DiskKv::open(&path, key, 1 << 20).unwrap();
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
            let mut d = DiskKv::open(&path, key, 1 << 20).unwrap();
            assert!(d.contains(1));
            assert_eq!(d.meta(1).unwrap().as_deref(), Some(&b"one"[..]));
            let mut out = KvRows::default();
            assert!(d.get(1, 0..2, &mut out).unwrap());
            assert_eq!(out.layers[0].k, vec![1; 8]);
            assert_eq!(d.extents().unwrap(), vec![1]);
        }
        // Another model's key sees none of it.
        let mut other = DiskKv::open(&path, key + 1, 1 << 20).unwrap();
        assert!(other.extents().unwrap().is_empty());
    }

    #[test]
    fn a_journal_file_is_not_mistaken_for_a_database() {
        let dir = temp_dir();
        let path = dir.path().join("kv.journal");
        std::fs::write(&path, b"INFRKV02 not a redb database").unwrap();
        assert!(DiskKv::open(&path, 1, 1 << 20).is_err());
    }

    #[test]
    fn a_disk_store_takes_its_lru_in_order() {
        let dir = temp_dir();
        let path = dir.path().join("kv.redb");
        let mut d = DiskKv::open(&path, 1, 1 << 20).unwrap();
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
        let mut t = Tiered::new(RamKv::new(8), RamKv::default());
        let a = KvRows::whole(1, vec![vec![1; 4]], vec![vec![2; 2]]);
        let b = KvRows::whole(1, vec![vec![3; 4]], vec![vec![4; 2]]);
        t.put(1, 0, a.clone()).unwrap();
        t.put(2, 0, b.clone()).unwrap();
        assert!(!t.fast().contains(1), "the older extent left RAM");
        assert!(t.slow().contains(1), "and landed in the slow tier");
        assert!(t.fast().contains(2));
        assert_eq!(t.fast().host(2).unwrap().layers, b.layers);
        assert_eq!(t.slow().host(1).unwrap().layers, a.layers);

        let mut got = KvRows::default();
        assert!(t.get(1, 0..1, &mut got).unwrap());
        assert_eq!(got.layers, a.layers);
        assert!(t.fast().contains(1), "a read promotes the extent");
        assert!(t.slow().contains(1), "the slow copy stays");
        assert_eq!(t.fast().host(1).unwrap().layers, a.layers);
        assert_eq!(t.slow().host(1).unwrap().layers, a.layers);
    }

    #[test]
    fn tiered_keeps_the_only_extent_even_over_budget() {
        let mut t = Tiered::new(RamKv::new(4), RamKv::default());
        let a = KvRows::whole(1, vec![vec![1; 4]], vec![vec![2; 2]]);
        t.put(1, 0, a.clone()).unwrap();
        assert!(t.fast().contains(1));
        assert!(!t.slow().contains(1));
    }

    /// A promotion must not leave the fast tier over budget: the read that just happened stays,
    /// and whatever became least recently used goes down.
    #[test]
    fn a_promotion_settles_the_fast_tier() {
        let mut t = Tiered::new(RamKv::new(8), RamKv::default());
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

    /// A block rewritten while an older copy sits below it must not leave that copy behind: the
    /// next start would load the stale rows, snapshot flag and all.
    #[test]
    fn a_rewritten_extent_drops_the_stale_copy_below_it() {
        let mut t = Tiered::new(RamKv::new(8), RamKv::default());
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
        let mut b = Bounded::new(RamKv::new(8));
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

    #[test]
    fn a_linear_slot_round_trips_through_the_cache() {
        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(0);
        let src = filled_slot(&be, &cfg, false, 300);
        let mut s = Cache::new(RamKv::default(), true);
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
        let mut s = Cache::new(RamKv::default(), true);
        assert_eq!(s.capture(&be, &cfg, &src, n).unwrap(), n);
        let ids = s.matched(&src.cached).1;
        assert_eq!(ids.len(), 6);
        // The window starts at row 444: blocks below it are empty, the one straddling it keeps
        // the rows from 444 on, and the rest are whole.
        for (j, id) in ids.iter().enumerate() {
            let rows = s.store.host(*id).unwrap();
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

    /// A restore whose blocks cannot fill the destination's window re-prefills instead of leaving
    /// rows attention will read unwritten.
    #[test]
    fn a_restore_into_a_wider_ring_refuses() {
        let be = infr_cpu::CpuBackend::new();
        let src_cfg = slot_cfg(200);
        let n = 700;
        let src = filled_slot(&be, &src_cfg, true, n);
        let mut s = Cache::new(RamKv::default(), true);
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
        let mut s = Cache::new(RamKv::default(), true);
        assert_eq!(s.capture(&be, &cfg, &src, 600).unwrap(), 600);
        let ids = s.matched(&src.cached).1;
        assert_eq!(ids.len(), 5);
        // Rows below 544 are gone: the block over 512..640 keeps only 544..600.
        let rows = s.store.host(ids[4]).unwrap();
        assert_eq!(rows.layers[0].runs, vec![32..88]);
    }

    /// One block of the two-layer test slot, in bytes.
    fn block_bytes(cfg: &Config) -> usize {
        cfg.n_layer * BLOCK_TOKENS * ROW * 2 * 2
    }

    type DiskCache = BlockCache<Tiered<RamKv, DiskKv>>;

    fn disk_cache(path: &Path, cap: usize, key: u64) -> DiskCache {
        let disk = DiskKv::open(path, key, 1 << 20).unwrap();
        BlockCache::new(Tiered::new(RamKv::new(cap), disk), true)
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
            let mut s = disk_cache(&path, cap, key);
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
            let mut s = disk_cache(&path, cap, key);
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
        use crate::seam::kv_tier::DISK_EXTENTS;

        let be = infr_cpu::CpuBackend::new();
        let cfg = slot_cfg(0);
        let dir = temp_dir();
        let path = dir.path().join("kv.redb");
        let key = 0x77u64;
        let src = filled_slot(&be, &cfg, false, 300);
        let cap = block_bytes(&cfg);
        let first = {
            let mut s = disk_cache(&path, cap, key);
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
                let mut table = txn.open_table(DISK_EXTENTS).unwrap();
                table.insert((key, first), b"not rows".as_slice()).unwrap();
            }
            txn.commit().unwrap();
        }

        let mut s = disk_cache(&path, cap, key);
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
        let mut s: BlockCache<Tiered<RamKv, RamKv>> =
            BlockCache::new(Tiered::new(RamKv::new(cap), RamKv::default()), true);
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
