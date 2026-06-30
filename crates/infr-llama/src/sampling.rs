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
impl Default for Sampler {
    fn default() -> Self {
        Self {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

/// A forward step's output: the sampled token (chosen on the GPU — only 4 bytes cross the bus) or
/// the full vocab logits (host samples them, when GPU sampling can't handle the config).
pub(crate) enum GenOut {
    Token(u32),
    Logits(Vec<f32>),
}

/// Per-step sampling config for on-GPU token selection. `u` is the host-drawn uniform in [0,1).
#[derive(Clone, Copy)]
pub(crate) struct SampleParams {
    pub(crate) temp: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) u: f32,
}
impl SampleParams {
    /// Greedy (argmax) when temperature is off or only one candidate is kept.
    pub(crate) fn greedy(&self) -> bool {
        self.temp <= 0.0 || self.top_k == 1
    }
    /// The GPU sampler handles temp/top-k/top-p only for a bounded top_k; else host samples logits.
    pub(crate) fn gpu_capable(&self) -> bool {
        !self.greedy() && self.top_k >= 2 && self.top_k <= infr_vulkan::Recorder::SAMPLE_KMAX
    }
}

/// Advance an xorshift64 RNG and return a uniform in [0,1) — the per-step random draw for sampling.
pub(crate) fn draw_u(rng: &mut u64) -> f32 {
    let mut x = *rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

/// Incremental UTF-8-safe detokenizer: fed the FULL decoded text each step, returns the newly
/// completed suffix. Byte-level BPE splits a multi-byte char (e.g. an emoji) across tokens, so a
/// step's decode can end mid-character as U+FFFD (`�`); we hold output until it completes (decode no
/// longer ends in `�`), emitting whole characters only.
#[derive(Default)]
pub(crate) struct StreamDecoder {
    printed: usize,
}
impl StreamDecoder {
    pub(crate) fn step(&mut self, full: &str) -> String {
        if !full.ends_with('\u{FFFD}')
            && full.len() > self.printed
            && full.is_char_boundary(self.printed)
        {
            let delta = full[self.printed..].to_string();
            self.printed = full.len();
            delta
        } else {
            String::new()
        }
    }
}

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
    // xorshift64 → uniform [0, total)
    let mut x = *rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    let u = (x >> 40) as f32 / (1u64 << 24) as f32;
    let r = u * total;
    let mut acc = 0.0;
    for j in 0..cutoff {
        acc += probs[j];
        if r <= acc {
            return idx[j] as u32;
        }
    }
    idx[cutoff - 1] as u32
}
