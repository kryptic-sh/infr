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

    /// The sampler the decode loop actually runs: a [`RequestSampling`] scope's EXPLICIT overrides
    /// (one `infr serve` HTTP request) layered over [`from_env`](Self::from_env). Outside a scope
    /// (`infr run`, `bench`, every test) this IS `from_env()` — byte-for-byte the old behavior.
    pub fn resolve() -> Self {
        let mut s = Self::from_env();
        with_request(|r| {
            if let Some(t) = r.temp {
                s.temp = t;
            }
            if let Some(k) = r.top_k {
                s.top_k = k;
            }
            if let Some(p) = r.top_p {
                s.top_p = p;
            }
        });
        s
    }
}

// ---------------------------------------------------------------------------
// Per-request sampling scope (the `infr serve` seam)
// ---------------------------------------------------------------------------

/// Per-request sampling overrides + penalty config + an abort latch, installed for the duration of
/// ONE generation by [`RequestScope`].
///
/// Why a thread-scoped value rather than an extra `generate_dense_backend` argument: sampling has to
/// reach the innermost decode step (penalties mutate the logits row; the stop-sequence matcher has
/// to *halt* the loop from inside the `on_piece` callback, which returns `()`), and that step sits
/// behind 6 backend wrappers and 18 call sites. Scoping it keeps ONE knob to thread instead of a
/// parameter on every path, and — critically — keeps the default path (`None`) a single
/// thread-local read per generation, not per token.
///
/// The scope is safe because generation is synchronous and single-threaded per request: `infr
/// serve` serialises generation behind a mutex on ONE `spawn_blocking` thread, and `on_piece` is
/// invoked inline from the decode loop on that same thread.
#[derive(Clone, Debug)]
pub struct RequestSampling {
    /// `None` = inherit the env/CLI default (this is what makes an ABSENT request field a no-op).
    pub temp: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    /// Per-request RNG seed. `None` = the usual `INFR_SEED`/wall-clock seed.
    pub seed: Option<u64>,
    /// OpenAI `presence_penalty` (-2..2): flat subtraction for any token already generated.
    pub presence_penalty: f32,
    /// OpenAI `frequency_penalty` (-2..2): subtraction scaled by the token's generated count.
    pub frequency_penalty: f32,
    /// llama.cpp `repeat_penalty` (1.0 = off): divides positive logits / multiplies negative ones
    /// for tokens seen in the last [`repeat_last_n`](Self::repeat_last_n) generated tokens.
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for RequestSampling {
    /// Neutral: every field inherits the env/CLI default, no penalty applied. `repeat_last_n`
    /// mirrors llama.cpp's default window (64) so a request that sets only `repeat_penalty` gets
    /// llama.cpp's behavior.
    fn default() -> Self {
        Self {
            temp: None,
            top_k: None,
            top_p: None,
            seed: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        }
    }
}

impl RequestSampling {
    /// Any penalty actually active? Drives the decode loop's host-sampling fallback (penalties must
    /// mutate the logits row, which the GPU-resident argmax/sample paths never download).
    pub fn penalties_active(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || (self.repeat_penalty != 1.0 && self.repeat_last_n > 0)
    }
}

thread_local! {
    static REQUEST: std::cell::RefCell<Option<RequestSampling>> = const { std::cell::RefCell::new(None) };
    /// Latched by [`request_abort`] from inside a streaming callback (the server's stop-sequence
    /// matcher); polled once per decoded token by the decode loop.
    static ABORT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn with_request<R>(f: impl FnOnce(&RequestSampling) -> R) -> Option<R> {
    REQUEST.with(|c| c.borrow().as_ref().map(f))
}

/// RAII guard installing a [`RequestSampling`] for the current thread. Restores the previous value
/// (normally `None`) and clears the abort latch on drop, so a panicking/erroring request can never
/// leak its sampling config into the next one on a reused `spawn_blocking` thread.
pub struct RequestScope(Option<RequestSampling>);

impl RequestScope {
    pub fn new(r: RequestSampling) -> Self {
        let prev = REQUEST.with(|c| c.borrow_mut().replace(r));
        ABORT.with(|c| c.set(false));
        Self(prev)
    }
}

impl Drop for RequestScope {
    fn drop(&mut self) {
        REQUEST.with(|c| *c.borrow_mut() = self.0.take());
        ABORT.with(|c| c.set(false));
    }
}

/// Ask the running decode loop to stop after the current token — the stop-sequence hit. Called from
/// inside the `on_piece` callback (which returns `()` and so has no other way to say "done").
pub fn request_abort() {
    ABORT.with(|c| c.set(true));
}

/// Polled by the decode loop once per token (a thread-local `Cell` read — no allocation, no lock).
pub(crate) fn abort_requested() -> bool {
    ABORT.with(|c| c.get())
}

/// The RNG seed for this generation: the request's explicit `seed` wins, else `INFR_SEED`, else
/// wall clock (see [`seed_rng`]).
pub(crate) fn resolve_seed() -> u64 {
    match with_request(|r| r.seed).flatten() {
        Some(s) => s | 1, // xorshift64 state must be nonzero
        None => seed_rng(),
    }
}

/// Repetition-penalty state for ONE generation. Allocated once per request (never per token) and
/// only when a penalty is actually non-neutral — [`resolve`](Self::resolve) returns `None`
/// otherwise, which is what keeps `infr run`/bench/tests on the untouched GPU-sampled hot path.
///
/// Cost per token is O(distinct generated tokens), NOT O(vocab): the counts map is walked to patch
/// just the logits of tokens that have actually been produced.
pub(crate) struct Penalties {
    presence: f32,
    frequency: f32,
    repeat: f32,
    last_n: usize,
    /// token id -> times generated so far (presence/frequency).
    counts: std::collections::HashMap<u32, u32>,
    /// The last `last_n` generated ids, in order (llama.cpp's `repeat_penalty` window).
    recent: std::collections::VecDeque<u32>,
}

impl Penalties {
    pub(crate) fn resolve() -> Option<Self> {
        let r = with_request(|r| r.clone())?;
        if !r.penalties_active() {
            return None;
        }
        Some(Self {
            presence: r.presence_penalty,
            frequency: r.frequency_penalty,
            repeat: r.repeat_penalty,
            last_n: r.repeat_last_n,
            counts: std::collections::HashMap::new(),
            recent: std::collections::VecDeque::new(),
        })
    }

    /// Patch `logits` in place for the tokens generated so far. Order matches llama.cpp's
    /// `penalties` sampler: repeat (multiplicative, sign-aware) then presence/frequency (additive).
    pub(crate) fn apply(&self, logits: &mut [f32]) {
        if self.repeat != 1.0 && self.last_n > 0 {
            for &t in &self.recent {
                let l = &mut logits[t as usize];
                *l = if *l > 0.0 {
                    *l / self.repeat
                } else {
                    *l * self.repeat
                };
            }
        }
        if self.presence != 0.0 || self.frequency != 0.0 {
            for (&t, &n) in &self.counts {
                logits[t as usize] -= self.presence + self.frequency * n as f32;
            }
        }
    }

    /// Record a token the loop just committed.
    pub(crate) fn observe(&mut self, t: u32) {
        *self.counts.entry(t).or_insert(0) += 1;
        if self.last_n > 0 {
            self.recent.push_back(t);
            while self.recent.len() > self.last_n {
                self.recent.pop_front();
            }
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
