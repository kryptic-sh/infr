//! Token sampling (greedy / temperature + top-k + top-p) and incremental UTF-8-safe
//! detokenization. Mechanically split out of `lib.rs` (no logic change).

/// Token sampling: greedy when `temp <= 0`, else temperature + top-k + top-p (nucleus). Qwen3
/// recommends temp 0.6 / top_k 20 / top_p 0.95 — pure greedy makes thinking models degenerate
/// (fail to close `</think>`, repeat, or stop without answering).
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Default for Sampler {
    fn default() -> Self {
        Self {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Sampler {
    /// Sampler from the INFR_TEMP / INFR_TOP_K / INFR_TOP_P env knobs — the seam paths' sampling
    /// config (the bespoke path plumbs the same values through `Llama::set_sampling`). Defaults to
    /// GREEDY when the vars are unset, so library callers and the golden/parity tests stay
    /// deterministic; the CLI sets chat-appropriate defaults for run/serve.
    pub fn from_env() -> Self {
        let f = |k: &str, d: f32| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let u = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        Self {
            temp: f("INFR_TEMP", 0.0),
            top_k: u("INFR_TOP_K", 20),
            top_p: f("INFR_TOP_P", 0.95),
        }
    }
}

/// RNG seed for a generation's sampling draws (unused under greedy). `INFR_SEED` pins it for
/// distribution-identity testing (chained vs per-token temp sampling must draw the same stream
/// given the same seed); unset falls back to a wall-clock seed.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn seed_rng() -> u64 {
    std::env::var("INFR_SEED")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E3779B97F4A7C15)
        })
        | 1
}

/// Advance the xorshift64 state and return a uniform draw in [0, 1) — the factored-out RNG step
/// shared by the host sampler and the GPU `Op::Sample` path (which uploads the draw as the
/// kernel's `u` input, keeping the two paths distribution-identical).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn next_uniform(rng: &mut u64) -> f32 {
    let mut x = *rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

/// Sample a token id from `logits` per `s`. Greedy if `temp<=0`/`top_k==1`; else temperature +
/// top-k + top-p (nucleus). `rng` is an xorshift64 state advanced in place.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_logits(logits: &[f32], s: Sampler, rng: &mut u64) -> u32 {
    if s.temp <= 0.0 || s.top_k == 1 {
        return argmax(logits) as u32;
    }
    let n = logits.len();
    let k = if s.top_k == 0 { n } else { s.top_k.min(n) };
    let cmp = |a: &usize, b: &usize| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    let mut idx: Vec<usize> = (0..n).collect();
    if k < n {
        idx.select_nth_unstable_by(k - 1, cmp); // top-k at the front (unordered)
        idx.truncate(k);
    }
    idx.sort_unstable_by(cmp); // descending by logit
    let maxl = logits[idx[0]];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - maxl) / s.temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    // nucleus: smallest prefix whose cumulative prob reaches top_p
    let mut cum = 0.0;
    let mut cutoff = probs.len();
    for (j, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= s.top_p {
            cutoff = j + 1;
            break;
        }
    }
    let total: f32 = probs[..cutoff].iter().sum();
    let r = next_uniform(rng) * total;
    let mut acc = 0.0;
    for j in 0..cutoff {
        acc += probs[j];
        if r <= acc {
            return idx[j] as u32;
        }
    }
    idx[cutoff - 1] as u32
}

/// Temperature + top-k + top-p (nucleus) truncated distribution over `logits`, returned as
/// `(vocab id, normalized probability)` pairs summing to 1 — the same support selection
/// [`sample_logits`] draws from, just not collapsed into a single draw. A fresh, SEPARATE
/// implementation (not a shared refactor of `sample_logits`) so the existing bit-pinned
/// greedy/temperature decode path is untouched by this addition — see `sample_logits`'s callers
/// (the GPU `Op::Sample` parity tests) for why that path's exact float ops must not move.
///
/// Used by the MTP temperature-aware speculative accept rule
/// (`crate::seam::model::spec_accept_stochastic`): the proposal (`q`, from the draft head) and
/// target (`p`, from the trunk verify) distributions are truncated with the SAME `Sampler` config,
/// so the importance ratio `p(x)/q(x)` and the residual `max(p-q,0)` are well-defined — a token
/// truncated out of a distribution simply has probability 0 in it.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn truncated_dist(logits: &[f32], s: Sampler) -> Vec<(u32, f32)> {
    let n = logits.len();
    let k = if s.top_k == 0 { n } else { s.top_k.min(n) };
    let cmp = |a: &usize, b: &usize| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    let mut idx: Vec<usize> = (0..n).collect();
    if k < n {
        idx.select_nth_unstable_by(k - 1, cmp);
        idx.truncate(k);
    }
    idx.sort_unstable_by(cmp); // descending by logit
    let maxl = logits[idx[0]];
    let temp = if s.temp > 0.0 { s.temp } else { 1.0 }; // this fn is only meaningful for temp>0
                                                        // callers; guard div-by-zero regardless
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - maxl) / temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let mut cum = 0.0;
    let mut cutoff = probs.len();
    for (j, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= s.top_p {
            cutoff = j + 1;
            break;
        }
    }
    let total: f32 = probs[..cutoff].iter().sum();
    idx[..cutoff]
        .iter()
        .zip(probs[..cutoff].iter())
        .map(|(&i, &p)| (i as u32, p / total))
        .collect()
}

/// Draw one id from a normalized `(id, prob)` distribution (as returned by [`truncated_dist`])
/// using the shared xorshift64 uniform draw — the stochastic MTP accept rule's residual/bonus
/// sampling (`crate::seam::model::spec_accept_stochastic`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_from_dist(dist: &[(u32, f32)], rng: &mut u64) -> u32 {
    let Some(&(last_id, _)) = dist.last() else {
        return 0; // empty distribution: nothing to draw (caller-guaranteed not to happen)
    };
    let r = next_uniform(rng);
    let mut acc = 0.0;
    for &(id, p) in dist {
        acc += p;
        if r <= acc {
            return id;
        }
    }
    last_id // float rounding: r landed a hair past the cumulative sum — take the last entry
}
