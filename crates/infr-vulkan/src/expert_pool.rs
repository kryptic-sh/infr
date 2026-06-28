//! Host-backed, LRU-cached VRAM pool for MoE expert weights.
//!
//! An MoE layer routes each token to a few experts out of many (e.g. top-8 of 128). Keeping *all*
//! experts resident costs O(n_experts) VRAM; most are cold at any moment. This pool keeps a bounded
//! set of VRAM **slots** and streams the active experts into them on demand, with LRU eviction so
//! hot experts stay resident. Total weight VRAM becomes `dense + n_slots·stride` regardless of how
//! many experts the model has.
//!
//! The consumer (a future MoE forward) holds the experts' weights host-side (mmap'd GGUF), and for
//! each routed expert calls [`ExpertPool::resident`] with that expert's bytes to get a VRAM buffer
//! to run its FFN matmuls against. One slot = one expert's packed weights (the caller decides the
//! internal layout, e.g. gate‖up‖down concatenated); the pool treats it as an opaque blob.
//!
//! Sizing note for the MoE plan: `n_slots` must be ≥ the experts processed concurrently within a
//! layer (top-k); a few extra slots raise the cache hit-rate across layers/tokens. Streaming is a
//! decode / limited-VRAM strategy — when all experts fit VRAM, load them resident and skip the pool.

use std::collections::HashMap;

use infr_core::{
    backend::{Buffer, BufferUsage},
    error::Result,
    Backend,
};

/// A bounded set of VRAM slots that cache MoE expert weights, streamed from host RAM on demand.
pub struct ExpertPool {
    /// `n_slots` device buffers, each `stride` bytes — one resident expert apiece.
    slots: Vec<Box<dyn Buffer>>,
    stride: usize,
    /// slot → the expert id currently resident in it (None = empty).
    resident_id: Vec<Option<usize>>,
    /// expert id → slot index (the inverse of `resident_id`, for O(1) hit lookup).
    slot_of: HashMap<usize, usize>,
    /// slot → last-touched tick (for LRU eviction).
    last_used: Vec<u64>,
    tick: u64,
    /// Residency stats (cache hits / misses-with-upload), for tuning `n_slots`.
    pub hits: u64,
    pub misses: u64,
}

impl ExpertPool {
    /// Allocate a pool of `n_slots` VRAM slots, each `stride` bytes (the max packed size of one
    /// expert's weights). VRAM cost is `n_slots * stride`, independent of the model's expert count.
    pub fn new(be: &dyn Backend, stride: usize, n_slots: usize) -> Result<Self> {
        assert!(
            n_slots > 0 && stride > 0,
            "expert pool needs >0 slots and stride"
        );
        let mut slots = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            slots.push(be.alloc(stride, BufferUsage::Weights)?);
        }
        Ok(Self {
            slots,
            stride,
            resident_id: vec![None; n_slots],
            slot_of: HashMap::new(),
            last_used: vec![0; n_slots],
            tick: 0,
            hits: 0,
            misses: 0,
        })
    }

    /// Number of slots (max experts resident at once).
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Ensure expert `id` is resident and return its VRAM buffer. On a cache hit the buffer is
    /// returned directly (and marked most-recently-used); on a miss the least-recently-used slot is
    /// evicted and `data` (that expert's packed host bytes, `len <= stride`) is uploaded into it.
    pub fn resident(&mut self, be: &dyn Backend, id: usize, data: &[u8]) -> Result<&dyn Buffer> {
        assert!(
            data.len() <= self.stride,
            "expert {id}: {} bytes exceeds slot stride {}",
            data.len(),
            self.stride
        );
        self.tick += 1;
        if let Some(&slot) = self.slot_of.get(&id) {
            self.last_used[slot] = self.tick;
            self.hits += 1;
            return Ok(self.slots[slot].as_ref());
        }
        // Miss: evict the least-recently-used slot (empty slots have tick 0 → picked first).
        self.misses += 1;
        let slot = (0..self.slots.len())
            .min_by_key(|&s| self.last_used[s])
            .expect("pool has >0 slots");
        if let Some(old) = self.resident_id[slot].take() {
            self.slot_of.remove(&old);
        }
        be.upload(self.slots[slot].as_ref(), data)?;
        self.resident_id[slot] = Some(id);
        self.slot_of.insert(id, slot);
        self.last_used[slot] = self.tick;
        Ok(self.slots[slot].as_ref())
    }

    /// True if expert `id` is currently resident (no upload would be needed).
    pub fn is_resident(&self, id: usize) -> bool {
        self.slot_of.contains_key(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VulkanBackend;

    fn expert_bytes(id: usize, len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_add((id as u8).wrapping_mul(37)))
            .collect()
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn pool_residency_lru_and_roundtrip() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let stride = 4096usize;
        let mut pool = ExpertPool::new(&be, stride, 2).expect("pool");
        assert_eq!(pool.capacity(), 2);

        // round-trip: the slot holds exactly the expert's uploaded bytes.
        let check = |be: &VulkanBackend, buf: &dyn Buffer, want: &[u8]| {
            let mut back = vec![0u8; want.len()];
            be.download(buf, &mut back).unwrap();
            assert_eq!(&back, want);
        };
        let e0 = expert_bytes(0, stride);
        let e1 = expert_bytes(1, stride);
        let e2 = expert_bytes(2, stride);

        let b0 = pool.resident(&be, 0, &e0).unwrap();
        check(&be, b0, &e0);
        let b1 = pool.resident(&be, 1, &e1).unwrap();
        check(&be, b1, &e1);
        assert_eq!((pool.hits, pool.misses), (0, 2));

        // hit: expert 0 still resident → no upload, marks 0 MRU.
        let _ = pool.resident(&be, 0, &e0).unwrap();
        assert_eq!((pool.hits, pool.misses), (1, 2));
        assert!(pool.is_resident(0) && pool.is_resident(1));

        // miss: expert 2 evicts the LRU slot. 0 was just touched (MRU), so 1 is evicted.
        let b2 = pool.resident(&be, 2, &e2).unwrap();
        check(&be, b2, &e2);
        assert!(pool.is_resident(2) && pool.is_resident(0));
        assert!(
            !pool.is_resident(1),
            "LRU expert 1 should have been evicted, not 0"
        );
        assert_eq!((pool.hits, pool.misses), (1, 3));

        // expert 0 still holds its bytes after the eviction churn.
        let b0b = pool.resident(&be, 0, &e0).unwrap();
        check(&be, b0b, &e0);
        assert_eq!((pool.hits, pool.misses), (2, 3));
    }
}
