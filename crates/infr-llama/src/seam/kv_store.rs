//! Cross-conversation KV prefix store.
//!
//! A prompt's K/V rows never change once written, so the rows one request materialized can seed
//! any later request that starts with the same tokens — a new conversation, an evicted slot, or a
//! different chat on the same model. The slot pool already reuses KV between its own slots
//! (`ParallelSeam::checkout` seeds from a free slot); this store is the same idea with the rows in
//! host RAM, so the reuse survives eviction and outlives the conversation that produced it.
//!
//! The payoff is the prefill it skips: 8k of KV uploads over PCIe in about 0.12 s, while
//! re-prefilling the same tokens costs tens of seconds. The cost is one download per request that
//! captures a prefix, so capture is gated to prefixes long enough to be worth re-reading.

use crate::Config;
use crate::EngineConfig;
use anyhow::Result;
use infr_core::backend::Backend;
use infr_core::budget::kv_fmt_bytes;
use infr_core::DType;

use super::{common_prefix_len, kv_row_elems, kv_rows, weights::SeamKv};

/// Below this many shared tokens the upload costs more than re-prefilling them.
pub(crate) const MIN_PREFIX_TOKENS: usize = 256;

/// One captured prefix: the token ids it covers and the raw K/V rows for `[0, tokens.len())`.
///
/// Rows are position-bound — attention applies rope before the row is stored — so an entry is only
/// good from position 0. That is what a conversation prefix is, so nothing is lost.
struct Entry {
    tokens: Vec<u32>,
    /// Per layer, in `SeamKv::kbufs` order. `v[l]` is empty for an MLA layer (no V side).
    k: Vec<Vec<u8>>,
    v: Vec<Vec<u8>>,
    bytes: usize,
    used: u64,
}

/// Host-RAM LRU store of captured prompt prefixes.
pub(crate) struct KvPrefixStore {
    entries: Vec<Entry>,
    cap_bytes: usize,
    /// Monotonic access counter (LRU order, not wall time — the store never sleeps).
    clock: u64,
    /// The KV layouts the entries were captured with; a dtype change invalidates every entry.
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
            entries: Vec::new(),
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

    /// The longest entry prefix shared with `prompt`, as `(entry index, tokens shared)`.
    fn best(&mut self, prompt: &[u32]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let p = common_prefix_len(&e.tokens, prompt).min(e.tokens.len());
            if p < MIN_PREFIX_TOKENS {
                continue;
            }
            if best.is_none_or(|(_, bp)| p > bp) {
                best = Some((i, p));
            }
        }
        if let Some((i, _)) = best {
            self.clock += 1;
            self.entries[i].used = self.clock;
        }
        best
    }

    /// Copy the longest stored prefix of `prompt` into `kv` (a device upload per layer) and mark
    /// those tokens materialized. Returns the number of tokens restored.
    ///
    /// The destination must be a linear cache: a wrapped SWA ring has recycled the early rows an
    /// entry holds. An unwrapped ring is fine up to its row count, so the restore stops there.
    pub(crate) fn restore(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        ec: &EngineConfig,
        kv: &mut SeamKv,
        prompt: &[u32],
    ) -> Result<usize> {
        if !self.enabled() || cfg.qwen35 {
            return Ok(0);
        }
        let Some((i, mut p)) = self.best(prompt) else {
            return Ok(0);
        };
        if self.k_fmt != kv.k_fmt || self.v_fmt != kv.v_fmt {
            return Ok(0);
        }
        p = p.min(kv.max_ctx);
        if kv.kv_ring {
            for l in 0..cfg.n_layer {
                if cfg.is_swa_layer(l) {
                    p = p.min(kv_rows(cfg, l, kv.max_ctx, true, ec));
                }
            }
        }
        if p < MIN_PREFIX_TOKENS {
            return Ok(0);
        }
        let e = &self.entries[i];
        for l in 0..cfg.n_layer {
            let (k_row, v_row) = kv_row_elems(cfg, l);
            let kb = kv_fmt_bytes(kv.k_fmt, p * k_row);
            be.upload(kv.kbufs[l].as_ref(), &e.k[l][..kb])?;
            if v_row > 0 {
                let vb = kv_fmt_bytes(kv.v_fmt, p * v_row);
                be.upload(kv.vbufs[l].as_ref(), &e.v[l][..vb])?;
            }
        }
        kv.cached = prompt[..p].to_vec();
        self.restores += 1;
        self.restored_tokens += p as u64;
        Ok(p)
    }

    /// Capture `kv`'s first `upto` rows as a reusable prefix. Returns the token count kept.
    ///
    /// Skipped when the rows are not reusable (a wrapped SWA ring keeps only its window, qwen35's
    /// recurrent state has no prefix), when the prefix is short, and when an entry already covers
    /// it — the last one is what keeps a repeated turn from re-downloading its own history.
    pub(crate) fn capture(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        ec: &EngineConfig,
        kv: &SeamKv,
        upto: usize,
    ) -> Result<usize> {
        if !self.enabled() || cfg.qwen35 {
            return Ok(0);
        }
        // A wrapped SWA ring has already recycled the rows an early prefix would need.
        if kv.kv_ring {
            for l in 0..cfg.n_layer {
                if cfg.is_swa_layer(l) && kv.cached.len() > kv_rows(cfg, l, kv.max_ctx, true, ec) {
                    return Ok(0);
                }
            }
        }
        let p = upto.min(kv.cached.len()).min(kv.max_ctx);
        if p < MIN_PREFIX_TOKENS {
            return Ok(0);
        }
        let tokens = &kv.cached[..p];
        if self
            .entries
            .iter()
            .any(|e| e.tokens.len() >= p && e.tokens[..p] == *tokens)
        {
            return Ok(0);
        }
        if self.k_fmt != kv.k_fmt || self.v_fmt != kv.v_fmt {
            self.entries.clear();
            self.k_fmt = kv.k_fmt;
            self.v_fmt = kv.v_fmt;
        }
        let mut k = Vec::with_capacity(cfg.n_layer);
        let mut v = Vec::with_capacity(cfg.n_layer);
        let mut bytes = 0usize;
        for l in 0..cfg.n_layer {
            let (k_row, v_row) = kv_row_elems(cfg, l);
            let kb = kv_fmt_bytes(kv.k_fmt, p * k_row);
            let mut buf = vec![0u8; kb];
            be.download(kv.kbufs[l].as_ref(), &mut buf)?;
            bytes += kb;
            k.push(buf);
            if v_row > 0 {
                let vb = kv_fmt_bytes(kv.v_fmt, p * v_row);
                let mut buf = vec![0u8; vb];
                be.download(kv.vbufs[l].as_ref(), &mut buf)?;
                bytes += vb;
                v.push(buf);
            } else {
                v.push(Vec::new());
            }
        }
        self.clock += 1;
        self.entries.push(Entry {
            tokens: tokens.to_vec(),
            k,
            v,
            bytes,
            used: self.clock,
        });
        self.captures += 1;
        self.trim();
        Ok(p)
    }

    /// Drop least-recently-used entries until the store fits its budget.
    fn trim(&mut self) {
        let mut total: usize = self.entries.iter().map(|e| e.bytes).sum();
        while total > self.cap_bytes && !self.entries.is_empty() {
            let (i, _) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .expect("non-empty");
            total -= self.entries.remove(i).bytes;
        }
    }

    #[cfg(test)]
    fn tokens_of(&self, i: usize) -> &[u32] {
        &self.entries[i].tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(seed: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| seed * 1000 + i).collect()
    }

    fn entry(tokens: &[u32], bytes: usize, used: u64) -> Entry {
        Entry {
            tokens: tokens.to_vec(),
            k: Vec::new(),
            v: Vec::new(),
            bytes,
            used,
        }
    }

    #[test]
    fn best_prefix_picks_the_longest_shared_entry() {
        let a = tokens(1, 300);
        let b = tokens(2, 400);
        let mut s = KvPrefixStore::new(1 << 20);
        s.entries = vec![entry(&a, 1, 0), entry(&b, 1, 0)];
        let mut extended = a.clone();
        extended.push(7);
        assert_eq!(s.best(&extended), Some((0, 300)));
        assert_eq!(s.best(&b[..320]), Some((1, 320)));
        // Shared prefix shorter than the floor is not worth an upload.
        assert_eq!(s.best(&a[..64]), None);
    }

    #[test]
    fn best_touches_lru_order() {
        let a = tokens(1, 300);
        let b = tokens(2, 400);
        let mut s = KvPrefixStore::new(1 << 20);
        s.entries = vec![entry(&a, 1, 1), entry(&b, 1, 2)];
        s.clock = 2;
        let _ = s.best(&b[..350]);
        assert_eq!(s.entries[1].used, 3);
    }

    #[test]
    fn trim_drops_the_least_recently_used() {
        let mut s = KvPrefixStore::new(10);
        s.entries = vec![entry(&tokens(1, 300), 6, 1), entry(&tokens(2, 300), 6, 2)];
        s.trim();
        assert_eq!(s.entries.len(), 1);
        assert_eq!(s.tokens_of(0), &tokens(2, 300));
    }
}
