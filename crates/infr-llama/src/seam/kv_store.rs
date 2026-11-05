//! Cross-conversation KV store, keyed in fixed-size token blocks.
//!
//! A prompt's K/V rows never change once written, so the rows one request materialized can seed
//! any later request that starts with the same tokens — a new conversation, an evicted slot, or a
//! different chat on the same model. The slot pool already reuses KV between its own slots
//! (`ParallelSeam::checkout` seeds from a free slot); this store is the same idea with the rows in
//! host RAM, so the reuse survives eviction and outlives the conversation that produced it.
//!
//! Rows are stored per 128-token block, chained: a block's id hashes its predecessor's id with its
//! own tokens, so a block only matches at the position it was written at (rope makes rows
//! position-bound) and ten conversations sharing a system prompt keep ONE copy of it.
//!
//! The payoff is the prefill it skips: 8k of KV uploads over PCIe in about 0.12 s, while
//! re-prefilling the same tokens costs tens of seconds.

use crate::Config;
use crate::EngineConfig;
use anyhow::Result;
use infr_core::backend::Backend;
use infr_core::budget::kv_fmt_bytes;
use infr_core::DType;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::{kv_row_elems, kv_rows, weights::SeamKv};

/// Tokens per stored block. Smaller blocks share more (a 600-token system prompt with two
/// different tails stores each half once) at the price of one download per new block.
pub(crate) const BLOCK_TOKENS: usize = 128;

/// Below this many shared tokens the upload costs more than re-prefilling them.
pub(crate) const MIN_PREFIX_TOKENS: usize = 256;

/// Chain id of a block: the previous block's id mixed with this block's tokens.
///
/// A collision only costs a wasted comparison — every lookup also compares the token ids.
fn block_id(prev: u64, tokens: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    prev.hash(&mut h);
    tokens.hash(&mut h);
    h.finish()
}

/// One stored block: `len` token ids and the raw K/V rows for that token range.
///
/// `len` is `BLOCK_TOKENS` except for the tail of a captured prefix, which may be short; a short
/// block only ever matches at the end of a prompt.
struct Block {
    id: u64,
    len: usize,
    tokens: Vec<u32>,
    /// Per layer, in `SeamKv::kbufs` order. `v[l]` is empty for an MLA layer (no V side).
    k: Vec<Vec<u8>>,
    v: Vec<Vec<u8>>,
    bytes: usize,
    used: u64,
}

/// Host-RAM LRU store of captured prompt blocks.
pub(crate) struct KvPrefixStore {
    blocks: Vec<Block>,
    cap_bytes: usize,
    /// Monotonic access counter (LRU order, not wall time — the store never sleeps).
    clock: u64,
    /// The KV layouts the blocks were captured with; a dtype change invalidates every block.
    k_fmt: DType,
    v_fmt: DType,
    pub(crate) restores: u64,
    pub(crate) restored_tokens: u64,
    pub(crate) captures: u64,
}

impl KvPrefixStore {
    /// Build a store with `cap_bytes` of host memory. A zero cap is a disabled store: every call
    /// returns immediately, so the engine pays nothing for a feature nobody asked for.
    pub(crate) fn new(cap_bytes: usize) -> Self {
        Self {
            blocks: Vec::new(),
            cap_bytes,
            clock: 0,
            k_fmt: DType::F16,
            v_fmt: DType::F16,
            restores: 0,
            restored_tokens: 0,
            captures: 0,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.cap_bytes > 0
    }

    /// `(blocks, bytes)` currently resident, for the capture log.
    pub(crate) fn stats(&self) -> (usize, usize) {
        (self.blocks.len(), self.blocks.iter().map(|b| b.bytes).sum())
    }

    fn find(&self, id: u64, tokens: &[u32]) -> Option<usize> {
        self.blocks
            .iter()
            .position(|b| b.id == id && b.tokens == tokens)
    }

    /// Walk `prompt` block by block; return the tokens covered by the leading stored run and the
    /// blocks that cover them, with their use stamped.
    fn matched(&mut self, prompt: &[u32]) -> (usize, Vec<usize>) {
        let mut id = 0u64;
        let mut n = 0usize;
        let mut hits = Vec::new();
        while n < prompt.len() {
            let len = BLOCK_TOKENS.min(prompt.len() - n);
            let block = &prompt[n..n + len];
            id = block_id(id, block);
            let Some(i) = self.find(id, block) else { break };
            self.clock += 1;
            self.blocks[i].used = self.clock;
            hits.push(i);
            n += len;
            if len < BLOCK_TOKENS {
                break;
            }
        }
        (n, hits)
    }

    /// Copy the longest stored run of blocks into `kv` (a device upload per layer) and mark those
    /// tokens materialized. Returns the number of tokens restored.
    ///
    /// The destination must be a ring that has not wrapped: a wrapped SWA ring has recycled the
    /// early rows a block holds. An unwrapped ring is fine up to its row count.
    pub(crate) fn restore(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        ec: &EngineConfig,
        kv: &mut SeamKv,
        prompt: &[u32],
    ) -> Result<usize> {
        // A paged slot's rows live at its block inside the pool, not at row 0, and the pool already
        // shares prefixes between its slots — the host-RAM store has nothing to add there.
        if !self.enabled() || cfg.qwen35 || kv.is_paged() {
            return Ok(0);
        }
        let (mut p, hits) = self.matched(prompt);
        if p < MIN_PREFIX_TOKENS || self.k_fmt != kv.k_fmt || self.v_fmt != kv.v_fmt {
            return Ok(0);
        }
        p = p.min(kv.max_ctx);
        if kv.kv_ring {
            for l in 0..cfg.n_layer {
                if cfg.is_swa_layer(l) {
                    let rows = kv_rows(cfg, l, kv.max_ctx, true, ec);
                    if kv.cached.len() > rows {
                        return Ok(0);
                    }
                    p = p.min(rows);
                }
            }
        }
        let mut keep = Vec::new();
        let mut n = 0usize;
        for i in hits {
            let len = self.blocks[i].len;
            if n + len > p {
                break;
            }
            n += len;
            keep.push(i);
        }
        if n < MIN_PREFIX_TOKENS {
            return Ok(0);
        }
        for l in 0..cfg.n_layer {
            let (k_row, v_row) = kv_row_elems(cfg, l);
            let kb = kv_fmt_bytes(kv.k_fmt, n * k_row);
            let mut buf = Vec::with_capacity(kb);
            for i in &keep {
                buf.extend_from_slice(&self.blocks[*i].k[l]);
            }
            debug_assert_eq!(buf.len(), kb);
            be.upload(kv.kbufs[l].as_ref(), &buf)?;
            if v_row > 0 {
                let vb = kv_fmt_bytes(kv.v_fmt, n * v_row);
                let mut buf = Vec::with_capacity(vb);
                for i in &keep {
                    buf.extend_from_slice(&self.blocks[*i].v[l]);
                }
                debug_assert_eq!(buf.len(), vb);
                be.upload(kv.vbufs[l].as_ref(), &buf)?;
            }
        }
        kv.cached = prompt[..n].to_vec();
        self.restores += 1;
        self.restored_tokens += n as u64;
        Ok(n)
    }

    /// Capture `kv`'s first `upto` rows as reusable blocks. Returns the token count kept.
    ///
    /// Blocks already stored are skipped, so a conversation that extends its own prefix keeps only
    /// the new suffix. Skipped entirely when the rows are not reusable (qwen35's recurrent state
    /// has no prefix) or the prefix is short.
    pub(crate) fn capture(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        ec: &EngineConfig,
        kv: &SeamKv,
        upto: usize,
    ) -> Result<usize> {
        if !self.enabled() || cfg.qwen35 || kv.is_paged() {
            return Ok(0);
        }
        let p = upto.min(kv.cached.len()).min(kv.max_ctx);
        if p < MIN_PREFIX_TOKENS {
            return Ok(0);
        }
        if self.k_fmt != kv.k_fmt || self.v_fmt != kv.v_fmt {
            self.blocks.clear();
            self.k_fmt = kv.k_fmt;
            self.v_fmt = kv.v_fmt;
        }
        // A wrapped SWA ring has recycled every row below the window it still holds.
        let mut safe_start = 0usize;
        if kv.kv_ring {
            let mut min_rows = usize::MAX;
            for l in 0..cfg.n_layer {
                if cfg.is_swa_layer(l) {
                    min_rows = min_rows.min(kv_rows(cfg, l, kv.max_ctx, true, ec));
                }
            }
            if min_rows != usize::MAX {
                safe_start = kv.cached.len().saturating_sub(min_rows);
            }
        }
        let tokens = &kv.cached[..p];
        let mut id = 0u64;
        let mut start = 0usize;
        let mut missing: Vec<(usize, usize, u64)> = Vec::new();
        while start < p {
            let len = BLOCK_TOKENS.min(p - start);
            id = block_id(id, &tokens[start..start + len]);
            if start >= safe_start && self.find(id, &tokens[start..start + len]).is_none() {
                missing.push((start, len, id));
            }
            start += len;
        }
        if missing.is_empty() {
            return Ok(0);
        }
        let base = self.blocks.len();
        for (start, len, id) in &missing {
            self.clock += 1;
            self.blocks.push(Block {
                id: *id,
                len: *len,
                tokens: tokens[*start..*start + *len].to_vec(),
                k: vec![Vec::new(); cfg.n_layer],
                v: vec![Vec::new(); cfg.n_layer],
                bytes: 0,
                used: self.clock,
            });
        }
        let filled = self.fill_from_device(be, cfg, kv, p, &missing, base);
        if let Err(e) = filled {
            self.blocks.truncate(base);
            return Err(e);
        }
        self.captures += 1;
        self.trim();
        Ok(p)
    }

    /// Download each new block's rows into the blocks pushed at `base .. base + missing.len()`.
    fn fill_from_device(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        kv: &SeamKv,
        p: usize,
        missing: &[(usize, usize, u64)],
        base: usize,
    ) -> Result<()> {
        for l in 0..cfg.n_layer {
            let (k_row, v_row) = kv_row_elems(cfg, l);
            let mut kbuf = vec![0u8; kv_fmt_bytes(kv.k_fmt, p * k_row)];
            be.download(kv.kbufs[l].as_ref(), &mut kbuf)?;
            let vbuf = if v_row > 0 {
                let mut buf = vec![0u8; kv_fmt_bytes(kv.v_fmt, p * v_row)];
                be.download(kv.vbufs[l].as_ref(), &mut buf)?;
                buf
            } else {
                Vec::new()
            };
            for (j, (start, len, _)) in missing.iter().enumerate() {
                let ks = kv_fmt_bytes(kv.k_fmt, *start * k_row);
                let ke = ks + kv_fmt_bytes(kv.k_fmt, *len * k_row);
                let kslice = kbuf[ks..ke].to_vec();
                let vslice = if v_row > 0 {
                    let vs = kv_fmt_bytes(kv.v_fmt, *start * v_row);
                    let ve = vs + kv_fmt_bytes(kv.v_fmt, *len * v_row);
                    vbuf[vs..ve].to_vec()
                } else {
                    Vec::new()
                };
                let b = &mut self.blocks[base + j];
                b.bytes += kslice.len() + vslice.len();
                b.k[l] = kslice;
                b.v[l] = vslice;
            }
        }
        Ok(())
    }

    /// Drop least-recently-used blocks until the store fits its budget.
    fn trim(&mut self) {
        let mut total: usize = self.blocks.iter().map(|b| b.bytes).sum();
        while total > self.cap_bytes && !self.blocks.is_empty() {
            let (i, _) = self
                .blocks
                .iter()
                .enumerate()
                .min_by_key(|(_, b)| b.used)
                .expect("non-empty");
            total -= self.blocks.remove(i).bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(seed: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| seed * 1000 + i).collect()
    }

    /// Push the chain for `tokens` as if it had just been captured.
    fn seed(s: &mut KvPrefixStore, tokens: &[u32], bytes: usize) {
        let mut id = 0u64;
        let mut n = 0usize;
        while n < tokens.len() {
            let len = BLOCK_TOKENS.min(tokens.len() - n);
            let block = &tokens[n..n + len];
            id = block_id(id, block);
            s.clock += 1;
            s.blocks.push(Block {
                id,
                len,
                tokens: block.to_vec(),
                k: Vec::new(),
                v: Vec::new(),
                bytes,
                used: s.clock,
            });
            n += len;
        }
    }

    #[test]
    fn matched_walks_the_chain_and_stops_at_a_gap() {
        let a = tokens(1, 300);
        let mut s = KvPrefixStore::new(1 << 20);
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
    fn shared_blocks_are_stored_once() {
        let a = tokens(1, 300);
        let mut b = a[..256].to_vec();
        b.extend(tokens(2, 200));
        let mut s = KvPrefixStore::new(1 << 20);
        seed(&mut s, &a, 1);
        assert_eq!(s.blocks.len(), 3);
        // The second prompt shares two blocks and adds two of its own.
        assert_eq!(s.matched(&b).0, 256);
        seed(&mut s, &b[256..], 1);
        assert_eq!(s.blocks.len(), 5);
        assert_eq!(s.blocks[4].len, 72);
    }

    #[test]
    fn trim_drops_the_least_recently_used() {
        let mut s = KvPrefixStore::new(10);
        seed(&mut s, &tokens(1, 300), 6);
        seed(&mut s, &tokens(2, 300), 6);
        assert_eq!(s.blocks.len(), 6);
        s.trim();
        // One block fits; it is the most recently used, i.e. the second prompt's tail.
        assert_eq!(s.blocks.len(), 1);
        assert_eq!(s.blocks[0].tokens, tokens(2, 300)[256..]);
    }
}
