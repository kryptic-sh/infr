//! Generated pager traces: the access patterns the engine's sessions make, as code, so the
//! simulator and the replay benchmark run without a GPU or a model.
//!
//! Each shape follows a survey of the consumers (`docs/plan/06-pager-eviction.md`, "What the
//! survey found"): dense streaming is one `schedule` per weight group, each its own batch; an MoE
//! pool holds one block per (layer, expert) of one role, a batched prefill opens one batch per
//! layer and cold-touches every expert in it, and decode either touches a fully resident layer
//! as a whole (`touch_all_hits`: one group touch since the per-layer group landed, every expert
//! MRU-touched before it) or the routed experts only.
//!
//! Randomness is xorshift64 (Marsaglia's 13/7/17 triple, as the crate's other tests use), seeded
//! per shape, so a shape is the same trace on every run and machine.

use super::trace::{Call, PoolTrace};
use super::BlockId;

/// xorshift64 (G. Marsaglia, "Xorshift RNGs", 2003, the `<<13 >>7 <<17` triple).
#[derive(Debug, Clone)]
pub struct XorShift64(u64);

impl XorShift64 {
    /// A generator from `seed`; a zero seed (xorshift's one fixed point) is replaced.
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)`, from the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `0..n` (`n > 0`); the modulo bias is below 2^-40 for any `n` a pool can have.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Zipf(`s`) over `0..n`: rank `k` drawn with probability proportional to `1 / (k + 1)^s`.
#[derive(Debug, Clone)]
pub struct Zipf {
    cdf: Vec<f64>,
}

impl Zipf {
    pub fn new(n: usize, s: f64) -> Self {
        let mut acc = 0.0;
        let mut cdf: Vec<f64> = (0..n)
            .map(|k| {
                acc += 1.0 / ((k + 1) as f64).powf(s);
                acc
            })
            .collect();
        for c in &mut cdf {
            *c /= acc;
        }
        Self { cdf }
    }

    pub fn sample(&self, rng: &mut XorShift64) -> usize {
        let u = rng.next_f64();
        self.cdf
            .partition_point(|&c| c <= u)
            .min(self.cdf.len() - 1)
    }
}

/// Dense layer streaming: `n_blocks` streamed weight groups visited `0..n_blocks` in order every
/// pass, each through `schedule` in its own batch (`DensePagerSession::stage`).
pub fn cyclic_sweep(n_blocks: u32, n_slots: usize, passes: usize) -> PoolTrace {
    let mut t = PoolTrace::generated(
        format!("cyclic sweep {n_blocks} blocks x{passes}"),
        n_slots,
        n_blocks as usize,
    );
    for _ in 0..passes {
        for id in 0..n_blocks {
            t.calls.push(Call::BeginBatch);
            t.calls.push(Call::Schedule(id));
        }
    }
    t
}

/// One MoE pool's geometry: one block per (layer, expert) of one role, ids `layer * n_expert +
/// expert` as the sessions assign them (`ExpertSource::layer_base`).
#[derive(Debug, Clone, Copy)]
pub struct Moe {
    pub n_layers: u32,
    pub n_expert: u32,
    /// Experts routed per token (`n_expert_used`).
    pub top_k: u32,
    /// Zipf exponent of each layer's expert popularity — an assumption, not a measurement; the
    /// recorded traces are what say how skewed real routing is.
    pub zipf_s: f64,
}

impl Moe {
    /// Qwen3-30B-A3B: 48 layers of 128 experts, 8 routed.
    pub const QWEN3_30B_A3B: Moe = Moe {
        n_layers: 48,
        n_expert: 128,
        top_k: 8,
        zipf_s: 1.0,
    };

    pub fn n_blocks(&self) -> usize {
        (self.n_layers * self.n_expert) as usize
    }

    fn id(&self, layer: u32, expert: u32) -> BlockId {
        layer * self.n_expert + expert
    }

    /// One batched-prefill chunk: per layer, one batch cold-touching every expert
    /// (`stage_role` with `scan = true`).
    pub fn prefill_chunk(&self, calls: &mut Vec<Call>) {
        for layer in 0..self.n_layers {
            calls.push(Call::BeginBatch);
            for e in 0..self.n_expert {
                calls.push(Call::TouchCold(self.id(layer, e)));
            }
        }
    }

    /// One decode token with every layer resident, as `touch_all_hits` made it before it touched
    /// layers as groups: per layer, one batch MRU-touching every expert.
    pub fn decode_token_all(&self, calls: &mut Vec<Call>) {
        for layer in 0..self.n_layers {
            calls.push(Call::BeginBatch);
            for e in 0..self.n_expert {
                calls.push(Call::Touch(self.id(layer, e)));
            }
        }
    }

    /// One decode token with every layer resident, as `touch_all_hits` makes it: per layer, one
    /// batch touching the layer's experts as one group.
    pub fn decode_token_grouped(&self, calls: &mut Vec<Call>) {
        for layer in 0..self.n_layers {
            calls.push(Call::BeginBatch);
            calls.push(Call::TouchGroup(self.id(layer, 0), self.n_expert));
        }
    }

    /// Per-layer routing state for [`Self::decode_token_routed`]: each layer ranks its experts by
    /// its own random permutation, so the popular experts differ between layers.
    pub fn router(&self, seed: u64) -> Router {
        let mut rng = XorShift64::new(seed);
        let rank = (0..self.n_layers)
            .map(|_| {
                let mut perm: Vec<u32> = (0..self.n_expert).collect();
                // Fisher-Yates.
                for i in (1..perm.len()).rev() {
                    perm.swap(i, rng.below(i + 1));
                }
                perm
            })
            .collect();
        Router {
            rank,
            zipf: Zipf::new(self.n_expert as usize, self.zipf_s),
            rng,
        }
    }

    /// One decode token on the routed path: per layer, one batch MRU-touching the `top_k`
    /// distinct experts the router picks.
    pub fn decode_token_routed(&self, router: &mut Router, calls: &mut Vec<Call>) {
        let mut picked = Vec::with_capacity(self.top_k as usize);
        for layer in 0..self.n_layers {
            picked.clear();
            while picked.len() < self.top_k.min(self.n_expert) as usize {
                let e = router.rank[layer as usize][router.zipf.sample(&mut router.rng)];
                if !picked.contains(&e) {
                    picked.push(e);
                }
            }
            calls.push(Call::BeginBatch);
            for &e in &picked {
                calls.push(Call::Touch(self.id(layer, e)));
            }
        }
    }

    /// `chunks` batched-prefill chunks over a pool of `n_slots`.
    pub fn prefill(&self, n_slots: usize, chunks: usize) -> PoolTrace {
        let mut t = self.pool(format!("moe prefill x{chunks}"), n_slots);
        for _ in 0..chunks {
            self.prefill_chunk(&mut t.calls);
        }
        t
    }

    /// `tokens` decode tokens with every layer resident.
    pub fn decode_resident(&self, tokens: usize) -> PoolTrace {
        let mut t = self.pool(format!("moe decode resident x{tokens}"), self.n_blocks());
        for _ in 0..tokens {
            self.decode_token_all(&mut t.calls);
        }
        t
    }

    /// [`Self::decode_resident`] as the session makes it today: one token touching every expert
    /// (which admits them all, as the first token after a prefill would have), then `tokens`
    /// tokens touching each layer as a group.
    pub fn decode_resident_grouped(&self, tokens: usize) -> PoolTrace {
        let mut t = self.pool(
            format!("moe decode resident grouped 1+{tokens}"),
            self.n_blocks(),
        );
        self.decode_token_all(&mut t.calls);
        for _ in 0..tokens {
            self.decode_token_grouped(&mut t.calls);
        }
        t
    }

    /// `tokens` routed decode tokens over a pool of `n_slots`.
    pub fn decode_routed(&self, n_slots: usize, tokens: usize, seed: u64) -> PoolTrace {
        let mut t = self.pool(
            format!(
                "moe decode top-{} zipf {} x{tokens}",
                self.top_k, self.zipf_s
            ),
            n_slots,
        );
        let mut router = self.router(seed);
        for _ in 0..tokens {
            self.decode_token_routed(&mut router, &mut t.calls);
        }
        t
    }

    /// A generation: one prefill chunk, then `tokens` routed decode tokens — the sweep and the
    /// skewed hot set competing for one pool.
    pub fn session(&self, n_slots: usize, tokens: usize, seed: u64) -> PoolTrace {
        let mut t = self.pool(format!("moe prefill + decode x{tokens}"), n_slots);
        self.prefill_chunk(&mut t.calls);
        let mut router = self.router(seed);
        for _ in 0..tokens {
            self.decode_token_routed(&mut router, &mut t.calls);
        }
        t
    }

    fn pool(&self, what: String, n_slots: usize) -> PoolTrace {
        PoolTrace::generated(
            format!("{what}, {}x{} blocks", self.n_layers, self.n_expert),
            n_slots,
            self.n_blocks(),
        )
    }
}

/// Routing state for [`Moe::decode_token_routed`], from [`Moe::router`].
#[derive(Debug, Clone)]
pub struct Router {
    rank: Vec<Vec<u32>>,
    zipf: Zipf,
    rng: XorShift64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The xorshift64 13/7/17 sequence from seed 1 — computed by hand from the definition, so a
    /// transcription slip in the shifts changes these.
    #[test]
    fn xorshift_matches_its_definition() {
        let mut r = XorShift64::new(1);
        // x = 1: x ^= x<<13 -> 0x2001; x ^= x>>7 -> 0x2001 ^ 0x40 = 0x2041;
        // x ^= x<<17 -> 0x2041 ^ 0x4082_0000 = 0x4082_2041.
        assert_eq!(r.next_u64(), 0x4082_2041);
    }

    #[test]
    fn zipf_ranks_are_ordered_by_frequency() {
        let z = Zipf::new(16, 1.0);
        let mut rng = XorShift64::new(7);
        let mut hist = [0u32; 16];
        for _ in 0..200_000 {
            hist[z.sample(&mut rng)] += 1;
        }
        // P(rank 0) / P(rank 1) = 2 under s = 1.
        let ratio = hist[0] as f64 / hist[1] as f64;
        assert!((ratio - 2.0).abs() < 0.1, "ratio {ratio}");
        assert!(hist[0] > hist[7] && hist[7] > hist[15]);
    }

    #[test]
    fn shapes_have_the_call_counts_their_sessions_make() {
        let m = Moe::QWEN3_30B_A3B;
        // Per layer: one batch and n_expert touches.
        let per_pass = (m.n_layers * (m.n_expert + 1)) as usize;
        assert_eq!(m.prefill(2048, 2).calls.len(), 2 * per_pass);
        assert_eq!(m.decode_resident(3).calls.len(), 3 * per_pass);
        // A grouped token is one batch and one group touch per layer.
        assert_eq!(
            m.decode_resident_grouped(3).calls.len(),
            per_pass + 3 * 2 * m.n_layers as usize
        );
        let routed = m.decode_routed(2048, 4, 1);
        assert_eq!(
            routed.calls.len(),
            4 * (m.n_layers * (m.top_k + 1)) as usize
        );
        // Routed ids stay inside their layer, distinct within a batch.
        let mut batch: Vec<BlockId> = Vec::new();
        for c in &routed.calls {
            match *c {
                Call::BeginBatch => batch.clear(),
                Call::Touch(id) => {
                    assert!((id as usize) < m.n_blocks());
                    assert!(!batch.contains(&id), "duplicate expert {id} in a batch");
                    batch.push(id);
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(cyclic_sweep(40, 8, 3).calls.len(), 2 * 40 * 3);
        // Same seed, same trace.
        assert_eq!(m.decode_routed(2048, 4, 1), routed);
    }
}
