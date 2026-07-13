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

    /// The sampler the decode loop actually runs: ONE sequence's EXPLICIT overrides (its
    /// [`RequestCtx`], carried by the scheduler's slot) layered over [`from_env`](Self::from_env).
    /// `req: None` (`infr run`, `bench`, every test) IS `from_env()` — byte-for-byte the old
    /// behavior.
    pub fn resolve(req: Option<&RequestCtx>) -> Self {
        let mut s = Self::from_env();
        if let Some(r) = req.map(RequestCtx::sampling) {
            if let Some(t) = r.temp {
                s.temp = t;
            }
            if let Some(k) = r.top_k {
                s.top_k = k;
            }
            if let Some(p) = r.top_p {
                s.top_p = p;
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Per-request sampling scope (the `infr serve` seam)
// ---------------------------------------------------------------------------

/// Per-request sampling overrides + penalty config — the CONFIG half of a [`RequestCtx`].
///
/// Every field is an `Option`/neutral default whose "unset" meaning is *inherit the process
/// default*, so a request that sends nothing generates EXACTLY as `infr run`/`bench`/the goldens do.
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

/// EVERYTHING one in-flight sequence owns that is not its KV cache: its sampling config, its abort
/// latch, and its turn on the GPU.
///
/// **This used to be a `thread_local!`** (a `RequestSampling` installed by an RAII `RequestScope`),
/// which was only sound because one generation owned one thread: `infr serve` serialised ALL
/// generation behind a single mutex. The moment N sequences make progress concurrently that
/// invariant dies — a thread-local is per-THREAD, not per-SEQUENCE, so every sequence would read
/// whichever config was installed last. Request A's temperature would silently apply to request B.
///
/// So the state is now EXPLICIT and per-sequence: the scheduler hands each in-flight sequence its
/// own `RequestCtx`, threaded into `generate_dense_backend` as `req: Option<&RequestCtx>` and read
/// nowhere else. `None` (`infr run`, `bench`, every test, every golden) means "inherit the process
/// default", i.e. byte-for-byte the pre-existing behavior — there is no thread-local left on any
/// path a decode step can reach.
///
/// Shared across threads by `&`: `abort` is an atomic and `gate` is an `Arc`, so the server's
/// `on_piece` callback can latch a stop-sequence hit on the same `&RequestCtx` the decode loop is
/// reading.
pub struct RequestCtx {
    sampling: RequestSampling,
    /// Latched by [`abort`](Self::abort) from inside a streaming callback (the server's
    /// stop-sequence matcher); polled once per decoded token by the decode loop.
    abort: std::sync::atomic::AtomicBool,
    /// This sequence's turn-taking baton on the GPU (`None` = sole user, e.g. `infr run`).
    gate: Option<std::sync::Arc<StepGate>>,
}

impl RequestCtx {
    /// A sequence with no GPU contention (a lone request, or a `-np 1` server).
    pub fn new(sampling: RequestSampling) -> Self {
        Self {
            sampling,
            abort: std::sync::atomic::AtomicBool::new(false),
            gate: None,
        }
    }

    /// A sequence sharing the GPU with the other slots of a `-np N` server: every decode step and
    /// prefill chunk takes a turn on `gate`.
    pub fn with_gate(sampling: RequestSampling, gate: std::sync::Arc<StepGate>) -> Self {
        Self {
            sampling,
            abort: std::sync::atomic::AtomicBool::new(false),
            gate: Some(gate),
        }
    }

    pub fn sampling(&self) -> &RequestSampling {
        &self.sampling
    }

    /// Ask the running decode loop to stop after the current token — the stop-sequence hit. Called
    /// from inside the `on_piece` callback (which returns `()` and so has no other way to say
    /// "done"). `Relaxed` is enough: the decode loop polls the SAME atomic and the callback runs
    /// inline on the decode thread; there is no other data to publish.
    pub fn abort(&self) {
        self.abort.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Polled by the decode loop once per token (one relaxed atomic load — no allocation, no lock).
    pub(crate) fn aborted(&self) -> bool {
        self.abort.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Take this sequence's turn on the GPU, blocking until the baton reaches it. `None` (no gate)
    /// is the uncontended path — the pass is never constructed, so a lone request pays NOTHING.
    pub(crate) fn gate_pass(&self) -> Option<GatePass<'_>> {
        self.gate.as_deref().map(StepGate::enter)
    }

    /// Is this sequence sharing the GPU? (A PREDICATE — unlike [`gate_pass`](Self::gate_pass) it
    /// does not take the baton.) Used to pick the prefill chunk size: a shared GPU wants small
    /// chunks so decodes aren't starved, a sole one wants big chunks for prefill throughput.
    pub(crate) fn shares_gpu(&self) -> bool {
        self.gate.is_some()
    }
}

/// The decode loop's abort poll, hoisted so `req: None` (every non-serve path) is a single
/// `Option::is_some_and` and not an atomic load.
pub(crate) fn abort_requested(req: Option<&RequestCtx>) -> bool {
    req.is_some_and(RequestCtx::aborted)
}

// ---------------------------------------------------------------------------
// The GPU baton (`infr serve --parallel N`)
// ---------------------------------------------------------------------------

/// A FAIR (FIFO) turnstile serialising GPU work across the N in-flight sequences of a `-np N`
/// server — the "one forward at a time, round-robin" rule.
///
/// It exists for two reasons, one hard and one soft:
///
/// 1. **Correctness.** `VulkanBackend` hands out its `VkCommandPool` by COPYING the handle out of
///    its mutex (`*cmd_pool.lock().unwrap()`) and then records/allocates outside the lock. Vulkan
///    requires a command pool be externally synchronised, so two threads recording concurrently is
///    UB. The baton is that external synchronisation.
/// 2. **Fairness.** A plain `Mutex` is not FIFO and can starve a waiter indefinitely. A ticket lock
///    hands the GPU to the longest-waiting sequence, so N clients round-robin at step granularity
///    and no request is head-of-line blocked behind another's whole generation.
///
/// The granularity is ONE decode step (or one chained decode chunk, or one prefill chunk) — see the
/// `gate_pass()` call sites in `seam::runner`. Uncontended cost is one mutex acquire/release per
/// step (~20ns against a multi-millisecond forward), and a `-np 1` server / `infr run` never
/// constructs a gate at all.
#[derive(Debug, Default)]
pub struct StepGate {
    inner: std::sync::Mutex<GateInner>,
    turn: std::sync::Condvar,
}

#[derive(Debug, Default)]
struct GateInner {
    /// Next ticket to hand out.
    next: u64,
    /// The ticket whose turn it is right now.
    serving: u64,
}

impl StepGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Block until this caller's ticket comes up. The returned [`GatePass`] releases the baton to
    /// the next ticket-holder on drop.
    fn enter(&self) -> GatePass<'_> {
        let mut g = self.inner.lock().expect("step gate poisoned");
        let ticket = g.next;
        g.next += 1;
        while g.serving != ticket {
            g = self.turn.wait(g).expect("step gate poisoned");
        }
        GatePass(self)
    }
}

/// The baton itself — held for exactly one GPU step, released on drop (including on error/panic,
/// which is why it is an RAII guard and not a pair of calls: a sequence that errors mid-step must
/// not wedge every other sequence forever).
pub(crate) struct GatePass<'a>(&'a StepGate);

impl Drop for GatePass<'_> {
    fn drop(&mut self) {
        // Poisoning: another sequence panicked mid-step. Advance anyway — refusing to would hang
        // every remaining client on a queue that can never drain.
        let mut g = match self.0.inner.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        g.serving += 1;
        drop(g);
        self.0.turn.notify_all();
    }
}

/// The RNG seed for this generation: the sequence's explicit `seed` wins, else `INFR_SEED`, else
/// wall clock (see [`seed_rng`]). Per-SEQUENCE, so `seed: 42` reproduces byte-identically no matter
/// how many other requests are in flight.
pub(crate) fn resolve_seed(req: Option<&RequestCtx>) -> u64 {
    match req.and_then(|r| r.sampling.seed) {
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
    pub(crate) fn resolve(req: Option<&RequestCtx>) -> Option<Self> {
        let r = req.map(RequestCtx::sampling)?;
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

// ---------------------------------------------------------------------------
// Tests — per-SEQUENCE isolation (the thread-local bug catcher)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(temp: f32, seed: u64) -> RequestSampling {
        RequestSampling {
            temp: Some(temp),
            seed: Some(seed),
            ..Default::default()
        }
    }

    /// **The regression test for the thread-local.**
    ///
    /// `RequestSampling` used to live in a `thread_local!` installed by an RAII `RequestScope`. That
    /// is per-THREAD, not per-SEQUENCE, so the instant ONE thread steps several sequences — exactly
    /// what a batched/interleaved scheduler does — every sequence reads whichever config was
    /// installed last. Request A's temperature would silently apply to request B, and no existing
    /// test would have caught it.
    ///
    /// So: interleave two sequences' sampling reads ON ONE THREAD (the batched-step shape) and
    /// demand each still sees its own config. Under the old design this test fails; under explicit
    /// per-sequence state it cannot.
    #[test]
    fn one_thread_stepping_two_sequences_keeps_their_sampling_separate() {
        let a = RequestCtx::new(cfg(0.0, 42));
        let b = RequestCtx::new(cfg(1.5, 7));

        for _ in 0..8 {
            // A step of sequence A, then a step of sequence B, then A again — one thread, both live.
            assert_eq!(Sampler::resolve(Some(&a)).temp, 0.0, "A must keep temp 0");
            assert_eq!(Sampler::resolve(Some(&b)).temp, 1.5, "B must keep temp 1.5");
        }
        assert_eq!(resolve_seed(Some(&a)), 42 | 1);
        assert_eq!(resolve_seed(Some(&b)), 7 | 1);

        // The abort latch (stop sequences) is per-sequence too: A hitting its stop string must not
        // halt B.
        a.abort();
        assert!(abort_requested(Some(&a)), "A latched its own abort");
        assert!(!abort_requested(Some(&b)), "B must NOT see A's abort");
        // And a non-serve caller (run/bench/tests/goldens) has no latch at all.
        assert!(!abort_requested(None));
    }

    /// `seed: 42` must reproduce byte-identically no matter how many other sequences are in flight.
    /// Each sequence carries its OWN xorshift state, so interleaving another sequence's draws
    /// between two of ours cannot perturb our stream.
    #[test]
    fn per_sequence_rng_is_reproducible_under_interleaving() {
        let logits: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let s = Sampler {
            temp: 1.0,
            top_k: 8,
            top_p: 0.95,
        };

        // Sequence A, alone.
        let a = RequestCtx::new(cfg(1.0, 42));
        let mut rng_a = resolve_seed(Some(&a));
        let alone: Vec<u32> = (0..16)
            .map(|_| sample_logits(&logits, s, &mut rng_a))
            .collect();

        // Sequence A again — but now a second sequence B draws from the SAME thread between every
        // one of A's draws (the interleaved-scheduler shape).
        let a2 = RequestCtx::new(cfg(1.0, 42));
        let b = RequestCtx::new(cfg(1.5, 7));
        let mut rng_a2 = resolve_seed(Some(&a2));
        let mut rng_b = resolve_seed(Some(&b));
        let interleaved: Vec<u32> = (0..16)
            .map(|_| {
                let t = sample_logits(&logits, s, &mut rng_a2);
                let _ = sample_logits(&logits, s, &mut rng_b); // B steps in between
                t
            })
            .collect();

        assert_eq!(
            alone, interleaved,
            "a seeded sequence must draw the same tokens whether or not it shares the engine"
        );
    }

    /// Penalties are per-sequence state (their token history is), and a sequence that sets none must
    /// stay on the untouched GPU-sampled hot path (`None`) even while another sequence has them on.
    #[test]
    fn penalties_are_per_sequence() {
        let plain = RequestCtx::new(RequestSampling::default());
        let penalized = RequestCtx::new(RequestSampling {
            repeat_penalty: 1.5,
            ..Default::default()
        });
        assert!(Penalties::resolve(Some(&plain)).is_none());
        assert!(Penalties::resolve(Some(&penalized)).is_some());
        assert!(Penalties::resolve(None).is_none());
    }

    /// The baton is mutually exclusive (only one sequence records on the GPU at a time) and FIFO
    /// (a waiter cannot be starved). Mutual exclusion is the CORRECTNESS property — the Vulkan
    /// command pool is externally synchronised.
    #[test]
    fn step_gate_is_mutually_exclusive_and_fifo() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let gate = std::sync::Arc::new(StepGate::new());
        let inside = std::sync::Arc::new(AtomicUsize::new(0));
        let max_seen = std::sync::Arc::new(AtomicUsize::new(0));
        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut hs = Vec::new();
        for id in 0..4 {
            let (gate, inside, max_seen, order) = (
                gate.clone(),
                inside.clone(),
                max_seen.clone(),
                order.clone(),
            );
            hs.push(std::thread::spawn(move || {
                for _ in 0..25 {
                    let _pass = gate.enter();
                    let n = inside.fetch_add(1, Ordering::SeqCst) + 1;
                    max_seen.fetch_max(n, Ordering::SeqCst);
                    order.lock().unwrap().push(id);
                    std::thread::yield_now();
                    inside.fetch_sub(1, Ordering::SeqCst);
                }
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            1,
            "two sequences were inside the gate at once — the GPU command pool would be racing"
        );
        assert_eq!(order.lock().unwrap().len(), 100);
    }
}
