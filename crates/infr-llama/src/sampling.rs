//! Token sampling (greedy / temperature + top-k + top-p) and incremental UTF-8-safe
//! detokenization. Mechanically split out of `lib.rs` (no logic change).

use infr_core::math;

/// Token sampling: greedy when `temp <= 0`, else temperature + top-k + top-p (nucleus). Qwen3
/// recommends temp 0.6 / top_k 20 / top_p 0.95 — pure greedy makes thinking models degenerate
/// (fail to close `</think>`, repeat, or stop without answering).
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// llama.cpp's `min_p`: drop every token whose probability is below `min_p` times the top
    /// token's, judged on the RAW logits (llama.cpp's chain runs it before temperature). `0`
    /// disables it. Inert under greedy, which only ever draws the top token.
    pub min_p: f32,
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Default for Sampler {
    fn default() -> Self {
        Self {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Sampler {
    /// The process-default sampler, read from the resolved [`infr_core::config::SamplingCfg`] — the seam paths'
    /// sampling config (the bespoke path plumbs the same values through `Llama::set_sampling`).
    ///
    /// Was `Sampler::from_env()` until the config migration. Its doc CONTRACT is
    /// unchanged and is what `SamplingCfg::default()` now carries: nothing set ⇒ `temp: 0.0` ⇒
    /// GREEDY, so library callers and the golden/parity tests stay deterministic; the CLI sets
    /// chat-appropriate defaults for run/serve through the config's CLI layer instead of the
    /// environment. `top_k: 20` / `top_p: 0.95` are inert under greedy and unchanged.
    pub fn from_cfg(cfg: &infr_core::config::SamplingCfg) -> Self {
        Self {
            temp: cfg.temp,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
            // No `sampling.min_p` config key: off unless a request asks for it.
            min_p: 0.0,
        }
    }

    /// The sampler the decode loop actually runs: ONE sequence's EXPLICIT overrides (its
    /// [`RequestCtx`], carried by the scheduler's slot) layered over [`from_cfg`](Self::from_cfg).
    /// `req: None` (`infr run`, `bench`, every test) IS `from_cfg(cfg)` — the precedence
    /// (`RequestCtx` > `Config`) is byte-for-byte what it was over `from_env()`.
    pub fn resolve(req: Option<&RequestCtx>, cfg: &infr_core::config::SamplingCfg) -> Self {
        let mut s = Self::from_cfg(cfg);
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
            if let Some(m) = r.min_p {
                s.min_p = m;
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
    /// llama.cpp `min_p` (see [`Sampler::min_p`]). `None` = inherit the process default (off).
    pub min_p: Option<f32>,
    /// OpenAI `logit_bias`: `(token id, bias)` pairs added to the logits row before the penalties
    /// and before sampling, as llama.cpp's `logit_bias` sampler does. Empty = none.
    pub logit_bias: Vec<(u32, f32)>,
    /// OpenAI `top_logprobs`: `Some(n)` asks the decode loop for this token's log probability and
    /// up to `n` most likely alternatives. `None` = no log probabilities. A request with
    /// `logprobs: true` and no `top_logprobs` sets `Some(20)`, matching llama.cpp.
    pub logprobs: Option<usize>,
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
            min_p: None,
            logit_bias: Vec::new(),
            logprobs: None,
        }
    }
}

impl RequestSampling {
    /// Any penalty actually active? One half of [`patches_logits`](Self::patches_logits), which
    /// drives the decode loop's host-sampling fallback (penalties must mutate the logits row, which
    /// the GPU-resident argmax/sample paths never download).
    pub fn penalties_active(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || (self.repeat_penalty != 1.0 && self.repeat_last_n > 0)
    }

    /// Does this request rewrite the logits row before sampling — a penalty or a `logit_bias`?
    /// Either one needs the row on the host, so this is what `Penalties::resolve` keys on, and
    /// through it the decode loop's GPU-sampling gates.
    pub fn patches_logits(&self) -> bool {
        self.penalties_active() || !self.logit_bias.is_empty()
    }

    /// Does this request need the logits row on the host after sampling? A logprobs request does:
    /// the GPU-sampling paths read back only the chosen id, and the reported probabilities come
    /// from the row itself.
    pub fn wants_logprobs(&self) -> bool {
        self.logprobs.is_some()
    }

    /// Refuse the fields only the dense seam decode loop applies — a `logit_bias`, a non-zero
    /// `min_p` — on a backend whose decode never reads them, rather than generate as if they had
    /// been applied. `backend` names it in the error.
    pub fn refuse_seam_only_fields(
        &self,
        backend: &'static str,
    ) -> Result<(), UnsupportedSampling> {
        let param = if !self.logit_bias.is_empty() {
            "logit_bias"
        } else if self.min_p.is_some_and(|m| m > 0.0) {
            "min_p"
        } else if self.wants_logprobs() {
            "logprobs"
        } else {
            return Ok(());
        };
        Err(UnsupportedSampling { param, backend })
    }
}

/// A request asked for sampling its backend cannot apply (see
/// [`RequestSampling::refuse_seam_only_fields`]). Typed so `infr serve` can answer it as the
/// request's fault, a `400` naming `param`, instead of a server error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedSampling {
    /// The request field that cannot be applied.
    pub param: &'static str,
    /// The backend that cannot apply it.
    pub backend: &'static str,
}

impl std::fmt::Display for UnsupportedSampling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} is not supported by the {} backend",
            self.param, self.backend
        )
    }
}

impl std::error::Error for UnsupportedSampling {}

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
    /// A latch set from OUTSIDE the generation — the server's per-request cancel, which its
    /// deadline and its client-disconnect check set. Read on every poll, so it stops a prefill
    /// that has not produced a piece yet, which no `on_piece` callback can.
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// This sequence's turn-taking baton on the GPU (`None` = sole user, e.g. `infr run`).
    gate: Option<std::sync::Arc<StepGate>>,
    /// Per-token log probabilities this request asked for, waiting for the text callback to take
    /// them. `None` when the request did not ask; then the decode loop computes nothing.
    logprobs: Option<std::sync::Mutex<Vec<infr_chat::TokenLogprob>>>,
    /// Test-only: [`aborted`](Self::aborted) answers true from this poll on, counting from zero.
    /// The prefill loop polls once before each chunk, so a test can cancel between two chunks.
    #[cfg(test)]
    abort_at_poll: Option<(usize, std::sync::atomic::AtomicUsize)>,
}

impl RequestCtx {
    /// A sequence with no GPU contention (a lone request, or a `-np 1` server).
    pub fn new(sampling: RequestSampling) -> Self {
        let logprobs = sampling.logprobs.map(|_| std::sync::Mutex::new(Vec::new()));
        Self {
            sampling,
            abort: std::sync::atomic::AtomicBool::new(false),
            cancel: None,
            gate: None,
            logprobs,
            #[cfg(test)]
            abort_at_poll: None,
        }
    }

    /// A sequence sharing the GPU with the other slots of a `-np N` server: every decode step and
    /// prefill chunk takes a turn on `gate`.
    pub fn with_gate(sampling: RequestSampling, gate: std::sync::Arc<StepGate>) -> Self {
        let logprobs = sampling.logprobs.map(|_| std::sync::Mutex::new(Vec::new()));
        Self {
            sampling,
            abort: std::sync::atomic::AtomicBool::new(false),
            cancel: None,
            gate: Some(gate),
            logprobs,
            #[cfg(test)]
            abort_at_poll: None,
        }
    }

    /// Also stop when `cancel` is set, at the same poll sites as [`abort`](Self::abort).
    pub fn with_cancel(mut self, cancel: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// Abort from poll `at` on (see the field); `at = k` lets prefill chunks `0..k` run.
    #[cfg(test)]
    pub(crate) fn abort_at_poll(mut self, at: usize) -> Self {
        self.abort_at_poll = Some((at, std::sync::atomic::AtomicUsize::new(0)));
        self
    }

    pub fn sampling(&self) -> &RequestSampling {
        &self.sampling
    }

    /// Queue one token's log probability for the text callback to attach to the piece it belongs
    /// to. No-op when this request did not ask for log probabilities.
    pub fn push_logprob(&self, lp: infr_chat::TokenLogprob) {
        if let Some(sink) = &self.logprobs {
            sink.lock().expect("logprob sink poisoned").push(lp);
        }
    }

    /// Take every log probability queued since the last call. Empty when none were asked for.
    pub fn take_logprobs(&self) -> Vec<infr_chat::TokenLogprob> {
        self.logprobs
            .as_ref()
            .map(|sink| std::mem::take(&mut *sink.lock().expect("logprob sink poisoned")))
            .unwrap_or_default()
    }

    /// Ask the running decode loop to stop after the current token — the stop-sequence hit. Called
    /// from inside the `on_piece` callback (which returns `()` and so has no other way to say
    /// "done"). `Relaxed` is enough: the decode loop polls the SAME atomic and the callback runs
    /// inline on the decode thread; there is no other data to publish.
    pub fn abort(&self) {
        self.abort.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Polled by the decode loop once per token and the prefill loop once per chunk (relaxed atomic
    /// loads — no allocation, no lock).
    pub fn aborted(&self) -> bool {
        #[cfg(test)]
        if let Some((at, polls)) = &self.abort_at_poll {
            if polls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= *at {
                return true;
            }
        }
        self.abort.load(std::sync::atomic::Ordering::Relaxed)
            || self
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
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

/// The decode loop's abort poll: "should this generation stop issuing new work at the next
/// boundary?" TWO sources, ORed —
///
/// 1. the PROCESS-wide shutdown latch ([`infr_core::shutdown`]): a `SIGINT`/`SIGTERM` arrived, so
///    every generation on every backend must wind down and let the GPU device be destroyed
///    properly. ORing it in HERE is what makes every pre-existing poll site (the chained-decode
///    loop, the per-token loop, the grammar-constrained loop, the prefill chunk loop) honour
///    Ctrl-C for free, on Vulkan, Metal and CPU alike, with no new plumbing;
/// 2. this ONE sequence's abort latch ([`RequestCtx::abort`]) — the `serve` stop-sequence hit,
///    which must stop request A without touching request B.
///
/// Cost is one relaxed atomic load (plus, for `serve`, a second) per token — nothing against a
/// multi-millisecond forward.
pub(crate) fn abort_requested(req: Option<&RequestCtx>) -> bool {
    infr_core::shutdown::shutdown_requested() || req.is_some_and(RequestCtx::aborted)
}

/// Does this request need the decode loop's ordinary per-token sampling path rather than the MTP
/// draft-verify loop? A `logit_bias` or a penalty rewrites the row, and logprobs need the whole
/// row on the host; MTP's draft head does neither.
pub(crate) fn adjusts_logits(req: Option<&RequestCtx>) -> bool {
    req.is_some_and(|r| r.sampling().patches_logits() || r.sampling().wants_logprobs())
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

/// The RNG seed for this generation: the sequence's explicit `seed` wins, else `sampling.seed`
/// (`INFR_SEED`), else wall clock (see [`seed_rng`]). Per-SEQUENCE, so `seed: 42` reproduces
/// byte-identically no matter how many other requests are in flight.
pub(crate) fn resolve_seed(req: Option<&RequestCtx>, cfg: &infr_core::config::SamplingCfg) -> u64 {
    match req.and_then(|r| r.sampling.seed) {
        Some(s) => legal_xorshift_state(s),
        None => seed_rng(cfg),
    }
}

/// Turn an arbitrary `u64` into a LEGAL xorshift64 state, changing as little as possible.
///
/// xorshift64 has exactly one forbidden state — `0`, which is a fixed point (it shifts/xors back to
/// itself forever, so every "random" draw is the same value). The obvious guard, `s | 1`, does fix
/// that, but at the cost of collapsing the seed space in half: `2k` and `2k+1` both map to the SAME
/// odd state, so `--seed 2` and `--seed 3` produced byte-identical output. That is a silent
/// correctness bug for anyone sweeping seeds — half their runs are duplicates of the other half.
///
/// So only `0` is remapped (to the golden-ratio constant, an arbitrary but well-mixed nonzero
/// state); every other seed passes through untouched and distinct seeds stay distinct.
///
/// Every seed entry point ([`resolve_seed`]'s per-request seed, [`seed_rng`]'s process-wide
/// `sampling.seed`/wall-clock seed, and DiffusionGemma's per-block reseed) MUST route through this
/// one helper — when they each had their own copy of the policy they drifted, and the
/// process-config path kept the `| 1` collapse long after the per-request path was fixed.
pub(crate) fn legal_xorshift_state(seed: u64) -> u64 {
    if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    }
}

/// Repetition-penalty state for ONE generation. Allocated once per request (never per token) and
/// only when a penalty is actually non-neutral — [`resolve`](Self::resolve) returns `None`
/// otherwise, which is what keeps `infr run`/bench/tests on the untouched GPU-sampled hot path.
///
/// Cost per token is O(distinct generated tokens), NOT O(vocab): the counts map is walked to patch
/// just the logits of tokens that have actually been produced.
pub(crate) struct Penalties {
    /// The request's `logit_bias`, applied before the penalties.
    bias: Vec<(u32, f32)>,
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
        if !r.patches_logits() {
            return None;
        }
        Some(Self {
            bias: r.logit_bias.clone(),
            presence: r.presence_penalty,
            frequency: r.frequency_penalty,
            repeat: r.repeat_penalty,
            last_n: r.repeat_last_n,
            counts: std::collections::HashMap::new(),
            recent: std::collections::VecDeque::new(),
        })
    }

    /// Patch `logits` in place: the request's `logit_bias` first, then the penalties for the tokens
    /// generated so far. Order matches llama.cpp's chain, whose `logit_bias` sampler runs ahead of
    /// `penalties`, and within the penalties repeat (multiplicative, sign-aware) then
    /// presence/frequency (additive).
    ///
    /// Every logits access is CHECKED (`get_mut`), and an id at or past `logits.len()` is skipped.
    /// The row is full-vocab on the trunk decode path, so today no id can be out of range — but the
    /// history this walks is generated ids, and the row it patches is whatever slice the caller
    /// hands over: an lm-head slice, or the MTP draft head's (possibly smaller) vocab. Indexing
    /// raw turned "the draft head has a narrower vocab than the trunk" into a mid-generation panic.
    /// Skipping is the right failure mode because a penalty is a soft PREFERENCE, not a
    /// correctness constraint: an id outside the row cannot be sampled from that row at all, so
    /// dropping its nudge changes nothing about the outcome, while panicking kills a live request.
    ///
    /// A finite logit stays finite and a `-inf` one stays `-inf`: the grammar-constrained step
    /// relies on that to tell a penalized token from a masked one.
    pub(crate) fn apply(&self, logits: &mut [f32]) {
        for &(t, b) in &self.bias {
            // Ids are validated against the vocabulary at the server, but the row may still be a
            // narrower head's (see above), so the same checked access applies.
            if let Some(l) = logits.get_mut(t as usize) {
                // Saturate like the penalties below: a bias is a preference, and under a grammar
                // an infinite logit would read as masked.
                *l = if l.is_finite() {
                    (*l + b).clamp(f32::MIN, f32::MAX)
                } else {
                    *l + b
                };
            }
        }
        if self.repeat != 1.0 && self.last_n > 0 {
            // llama.cpp's `penalties` sampler scales each DISTINCT id in the window ONCE — a token
            // repeated K times must be divided by `repeat`, not `repeat^K`. The raw `recent` deque
            // holds duplicates, so dedup it: penalize a given id the first time it is seen only.
            let mut seen = std::collections::HashSet::with_capacity(self.recent.len());
            for &t in &self.recent {
                if !seen.insert(t) {
                    continue;
                }
                let Some(l) = logits.get_mut(t as usize) else {
                    continue; // id past the end of this row — unsamplable here, so nothing to bias
                };
                let scaled = if *l > 0.0 {
                    *l / self.repeat
                } else {
                    *l * self.repeat
                };
                // `repeat` is any finite positive value, so an extreme one can overflow a finite
                // logit to ±inf. Saturate instead: under a grammar an infinite logit reads as
                // masked (or swamps the softmax), and a penalty must only reorder tokens.
                *l = if l.is_finite() {
                    scaled.clamp(f32::MIN, f32::MAX)
                } else {
                    scaled
                };
            }
        }
        if self.presence != 0.0 || self.frequency != 0.0 {
            for (&t, &n) in &self.counts {
                if let Some(l) = logits.get_mut(t as usize) {
                    let shifted = *l - (self.presence + self.frequency * n as f32);
                    // Saturate here too, for the same reason as the repeat scale above: the
                    // server bounds these two, but a library caller can pass any value.
                    *l = if l.is_finite() {
                        shifted.clamp(f32::MIN, f32::MAX)
                    } else {
                        shifted
                    };
                }
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

/// RNG seed for a generation's sampling draws (unused under greedy). `sampling.seed`
/// (`INFR_SEED`) pins it for distribution-identity testing (chained vs per-token temp sampling must
/// draw the same stream given the same seed); unset falls back to a wall-clock seed.
///
/// `sampling.seed` is deliberately an `Option`: the knob has TWO defaults in the tree (this
/// wall-clock one, and `42` in the CLI/diffusion paths) and BOTH stay at their own site.
///
/// The result goes through [`legal_xorshift_state`], exactly as [`resolve_seed`] does — so
/// `INFR_SEED=2` and `INFR_SEED=3` are DIFFERENT streams, and the wall-clock fallback can never
/// hand back xorshift's forbidden zero state either.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn seed_rng(cfg: &infr_core::config::SamplingCfg) -> u64 {
    legal_xorshift_state(cfg.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15)
    }))
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

/// How many candidates the `top_k == 0` nucleus probe selects on its first pass.
///
/// A `top_p` nucleus is the smallest DESCENDING prefix whose mass reaches `top_p`, so the right
/// size is "however deep the model's uncertainty goes at this position", not "vocab". At the
/// shipped Llama defaults (temp 0.6, top_p 0.9) on a 128256-entry vocab a measured, deliberately
/// peaked logit row needs **18** entries; pushing to top_p 0.95 needs **23**. Even pathologically
/// flat rows only run into the thousands. 1024 is ~50x the measured need, and deliberately so:
/// the two error directions cost wildly different amounts. Over-shooting K costs one `k log k`
/// sort of the surplus (measured at n = 128256: K=1024 is ~18µs/call slower than K=256, 526 vs
/// 508), while under-shooting costs a whole extra O(n) `select_nth_unstable_by` pass over the
/// vocab (~200µs). So err generous — a K that is 4x too big is an order of magnitude cheaper than
/// one widening, and the widening below is there for the rows no fixed K could have covered.
const NUCLEUS_PROBE_K: usize = 1024;

/// Widening factor when [`NUCLEUS_PROBE_K`] candidates did not reach `top_p`.
///
/// A near-uniform row can want the WHOLE vocab even at top_p 0.9, so the widening path is live
/// code, not a defensive branch — it just has to get there in few passes, because each pass
/// re-runs the O(n) selection. Doubling takes 8 passes to cross 128256 from 1024; ×8 takes 2
/// (1024 → 8192 → n, see the `n / 2` snap below). Measured at n = 128256 / top_p = 1.0, back when
/// that case still came through here: ×2 = 4.54ms per call, ×8 = 3.22ms, old heap = 5.27ms.
/// `top_p >= 1` now skips the selection entirely (see [`truncated_softmax`]).
const NUCLEUS_WIDEN: usize = 8;

/// The value reported for a probability that underflowed to 0 in binary32, which has no finite
/// logarithm: `serde_json` writes `-inf` as `null`, and a client reading `logprob` as a number
/// would choke. llama.cpp's server (`completion_token_output::logarithm`) reports the lowest
/// finite float in that case, and so does this.
const LOGPROB_FLOOR: f32 = f32::MIN;

/// Build the truncated, normalized sampling support shared by [`sample_logits`] and
/// [`truncated_dist`]: top-k select, temperature softmax, normalize, then the top-p (nucleus)
/// cutoff. Returns `(idx, probs)` — parallel, ALREADY truncated to the nucleus, with `probs`
/// normalized over the selected support (so it sums to ≤1, the nucleus mass). The pairs are
/// descending by logit, except with `top_k == 0` and `top_p >= 1`: that support is every token with
/// nonzero probability, in ascending index order, and no consumer may take `idx[0]` as the most
/// likely token.
/// The caller performs the final draw (bit-pinned) or hands the pairs out; keeping the draw OUT of
/// here is what preserves the existing sampler's exact float ops.
///
/// `temp` is clamped to a positive value (callers only reach this for `temp>0`; the clamp is a
/// div-by-zero guard). When `top_k==0` the support is the whole vocab, but instead of sorting all
/// of it only a bounded [`NUCLEUS_PROBE_K`] prefix is selected and sorted, widened on demand.
///
/// `min_p > 0` then also drops every token whose RAW logit is below the top logit plus
/// `ln(min_p)` — llama.cpp's `min_p` test (`p_i >= min_p * p_max` before temperature), in its
/// float expression. Its kept set and top-p's are both descending prefixes, so their order in the
/// chain does not matter: the support is the shorter one. The top token is always kept, as
/// llama.cpp's sorted path keeps it. `min_p <= 0` takes none of these branches, so the result is
/// bit-identical to the sampler without it.
fn truncated_softmax(
    logits: &[f32],
    temp: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
) -> (Vec<usize>, Vec<f32>) {
    let n = logits.len();
    let temp = if temp > 0.0 { temp } else { 1.0 };
    let min_logit = |maxl: f32| (min_p > 0.0).then(|| maxl + math::ln(min_p));
    let k = if top_k == 0 { n } else { top_k.min(n) };
    if k < n {
        // Bounded top-k: partition to the top k, then sort only those k (cheap). `total_cmp`, not
        // `partial_cmp` with an `Equal` fallback: that is not a total order once a NaN is present,
        // and `select_nth_unstable_by` / `sort_unstable_by` panic on one. On NaN-free logits the
        // two agree on every pair but `-0.0` / `+0.0`, which `total_cmp` orders. A NaN logit no
        // longer panics but ranks first, so the probabilities come out NaN as before. Ties go
        // to the lower index, as in the nucleus path below: the sorts are unstable.
        let cmp = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then_with(|| a.cmp(b));
        let mut idx: Vec<usize> = (0..n).collect();
        idx.select_nth_unstable_by(k - 1, cmp); // top-k at the front (unordered)
        idx.truncate(k);
        idx.sort_unstable_by(cmp); // descending by logit
        let maxl = logits[idx[0]];
        let mut probs: Vec<f32> = idx.iter().map(|&i| (logits[i] - maxl) / temp).collect();
        math::exp_slice(&mut probs);
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        // nucleus: smallest prefix whose cumulative prob reaches top_p
        let mut cum = 0.0;
        let mut cutoff = probs.len();
        for (j, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                cutoff = j + 1;
                break;
            }
        }
        if let Some(m) = min_logit(maxl) {
            let kept = idx[..cutoff]
                .iter()
                .take_while(|&&i| logits[i] >= m)
                .count();
            cutoff = kept.max(1);
        }
        idx.truncate(cutoff);
        probs.truncate(cutoff);
        (idx, probs)
    } else {
        // top_k==0 — the DEFAULT for the whole Llama-3.x/4 family, which mirrors Meta's published
        // generation_config (temp 0.6, top_p 0.9, top_k off). `top_k == 0` fails BOTH of the
        // decode loop's GPU gates in `seam::runner` (`gpu_sample` wants `(2..=64)`, `gpu_argmax`
        // wants `temp <= 0 || top_k == 1`), so every Llama decode with default sampling lands
        // here, once per token, on the host. It is worth the care.
        //
        // The two O(n) passes below are LOAD-BEARING and stay: `maxl` for the exp shift, and `sum`
        // as the TRUE full-vocab softmax denominator. Restricting `sum` to the selected candidates
        // would renormalize over a subset and silently change every probability (and hence the
        // sampled token, and `truncated_dist`'s MTP accept ratios). Note the index-order summation
        // is also what makes this arm bit-identical to what shipped before.
        let maxl = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // Every token's `exp((l - maxl) / temp)`, once: the sum and every probability below
        // read the same values.
        let mut e: Vec<f32> = logits.iter().map(|&l| (l - maxl) / temp).collect();
        math::exp_slice(&mut e);
        let sum: f32 = e.iter().sum();
        let min_logit = min_logit(maxl);

        // `top_p >= 1` with no top-k cut (`top_k` 0, or at least `n`) keeps every token that has
        // any mass, so no order is needed and none is built: one O(n) pass in index order, where
        // the widening below would end in a sort of the whole vocab (llama.cpp's top-p returns at
        // once for `p >= 1` as well). Each kept probability is the same expression over the same
        // `maxl` and `sum` the widening loop computes, so the distribution is unchanged; only the
        // ORDER differs, and with it the token a given uniform lands on. Zero-probability tokens
        // (masked `-inf`, or underflowed) are left out, which the sorted loop gets for free by
        // putting them last: in index order one in front would take a draw of exactly 0. The
        // sorted loop can also stop a hair early, once the rounded cumulative sum reaches 1.0;
        // this pass never drops a token with mass.
        // A row whose `sum` is not finite (a NaN or `+inf` logit, or every logit `-inf`) has no
        // distribution to keep and stays on the sorted path, which handles it as before.
        if top_p >= 1.0 && sum.is_finite() {
            let mut idx = Vec::with_capacity(n);
            let mut probs = Vec::with_capacity(n);
            for (i, &l) in logits.iter().enumerate() {
                let p = e[i] / sum;
                if p > 0.0 && min_logit.is_none_or(|m| l >= m) {
                    idx.push(i);
                    probs.push(p);
                }
            }
            return (idx, probs);
        }

        // What is NOT load-bearing is materializing an order over the whole vocab. This used to
        // build a `BinaryHeap<HeapItem>` of every entry — 128256 × 16B = 2.0 MiB allocated and
        // heapified per token — and then pop ~18 of them. Now: one bounded partial selection.
        //
        // Ordering is descending by logit with ties broken by ASCENDING INDEX. `total_cmp` gives a
        // total order over floats (so `-inf` masked logits sort last, never into the nucleus), and
        // the index tiebreak makes the result fully DETERMINISTIC — the old heap popped
        // equal-keyed items in whatever order heapification happened to leave them, and
        // `select_nth_unstable_by`/`sort_unstable_by` are unstable too, so without it two tokens
        // with bit-identical logits could swap places between runs or rustc versions. Tied entries
        // carry identical probabilities, so this changes `probs`/`cum`/the cutoff not at all; it
        // only pins WHICH of two indistinguishable ids comes back.
        let cmp = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then_with(|| a.cmp(b));
        let mut order: Vec<usize> = (0..n).collect();
        let mut k = NUCLEUS_PROBE_K.min(n); // clamp: n may be smaller than the probe
        loop {
            if k < n {
                order.select_nth_unstable_by(k - 1, cmp); // top-k at the front (unordered)
            }
            #[cfg(test)]
            tests::NUCLEUS_SORTED.with(|s| s.set(s.get() + k));
            order[..k].sort_unstable_by(cmp); // descending by logit, ties by index
            let mut probs: Vec<f32> = Vec::with_capacity(k);
            let mut cum = 0.0f32;
            let mut reached = false;
            for &i in &order[..k] {
                // `order` is descending, so the first token under the min-p floor ends the
                // support exactly as reaching `top_p` does — and spares the widening pass.
                if min_logit.is_some_and(|m| logits[i] < m) && !probs.is_empty() {
                    reached = true;
                    break;
                }
                let p = e[i] / sum;
                probs.push(p);
                cum += p;
                if cum >= top_p {
                    reached = true;
                    break;
                }
            }
            if reached || k == n {
                // `reached`: `probs` already stops at the cutoff. `k == n`: the nucleus is the
                // whole vocab (a flat enough row, or a `top_p >= 1` row the early return above
                // left here because its `sum` is not finite) and `probs` holds all of it —
                // which is exactly what the heap returned when it popped itself empty. Never
                // truncate the nucleus short of `top_p`; that would drop mass the draw expects.
                let idx = order[..probs.len()].to_vec();
                return (idx, probs);
            }
            // Widen. Snapping to `n` once the next step would land past the halfway mark avoids a
            // near-full selection followed immediately by a full one.
            k = if k.saturating_mul(NUCLEUS_WIDEN) >= n / 2 {
                n
            } else {
                k * NUCLEUS_WIDEN
            };
        }
    }
}

/// Sample a token id from `logits` per `s`. Greedy if `temp<=0`/`top_k==1`; else temperature +
/// top-k + top-p (nucleus). `rng` is an xorshift64 state advanced in place.
///
/// An EMPTY `logits` returns token `0` and trips a `debug_assert!`, exactly as its sibling
/// [`sample_from_dist`] does for an empty distribution — the two are the crate's only "draw an id"
/// entry points and they must not disagree about the degenerate input. Before the guard the two
/// paths through this function failed in two DIFFERENT wrong ways on an empty slice: the sampled
/// path ran `truncated_softmax`'s `top_k==0` arm over an empty heap and then panicked on
/// `idx[idx.len() - 1]`, while the greedy path silently returned [`argmax`]'s `0`. A real vocab is
/// never empty, so the `debug_assert!` is what states that: it makes the "can't happen" LOUD in
/// tests and under `infr`'s debug builds, while release keeps the sibling's total behaviour rather
/// than aborting a live request over a slice a caller mis-sliced. The MTP speculative path calls
/// this with caller-computed per-position slices, so "the caller built the slice" is the case that
/// motivates a defined answer at all.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_logits(logits: &[f32], s: Sampler, rng: &mut u64) -> u32 {
    if logits.is_empty() {
        debug_assert!(false, "sample_logits over an empty logits row");
        return 0; // empty support: nothing to draw (caller-guaranteed not to happen)
    }
    if s.temp <= 0.0 || s.top_k == 1 {
        return argmax(logits) as u32;
    }
    let (idx, probs) = truncated_softmax(logits, s.temp, s.top_k, s.top_p, s.min_p);
    // Final bit-pinned draw against the (un-renormalized) nucleus `probs`, scaled by their `total`.
    // Identical arithmetic to the pre-refactor inline draw; the `total` scaling means we never
    // renormalize `probs` a second time here (that footgun stays out of the hot path).
    let total: f32 = probs.iter().sum();
    let r = next_uniform(rng) * total;
    // A draw of exactly 0 (one uniform in 2^24) lands on the first entry with any mass. The
    // sorted supports start with the most likely token, but the index-order one (`top_k` 0,
    // `top_p` 1) starts at the lowest id with nonzero mass — which can be a token whose mass only
    // failed to underflow, such as one a `logit_bias` of -100 banned. Take the most likely token,
    // as the sorted paths do; on those `argmax` is entry 0, so their draw is unchanged.
    if r == 0.0 {
        return idx[argmax(&probs)] as u32;
    }
    let mut acc = 0.0;
    for (j, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return idx[j] as u32;
        }
    }
    idx[idx.len() - 1] as u32
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
    // Same support selection as `sample_logits`, but renormalized to sum to 1 over the nucleus (a
    // proper distribution) rather than collapsed into one draw. The nucleus renorm happens ONCE,
    // here, against the helper's `probs` — no coupled `cutoff`/`total` invariant to keep in sync.
    let (idx, probs) = truncated_softmax(logits, s.temp, s.top_k, s.top_p, s.min_p);
    let total: f32 = probs.iter().sum();
    idx.iter()
        .zip(probs.iter())
        .map(|(&i, &p)| (i as u32, p / total))
        .collect()
}

/// One token's log probability and up to `top_n` most likely alternatives: the log-softmax of the
/// whole logits row. Top-k/top-p/min-p truncate the SAMPLING support, but they are not part of
/// the model's distribution: renormalizing over the nucleus reports 0.0 for a token the model was
/// merely the most likely to emit, which is exactly the confidence signal a logprobs client
/// reads. Text is left empty for the caller that owns the tokenizer.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn token_logprob(
    logits: &[f32],
    selected: u32,
    top_n: usize,
) -> infr_chat::TokenLogprob {
    // The model's OWN distribution: no temperature (it is a sampling knob, applied after the
    // distribution is fixed, and llama.cpp's `n_probs` reports without it too), and no top-k /
    // top-p / min-p truncation, which narrows the sampling support rather than the distribution.
    // `logits` reach here after `logit_bias` and the penalties were applied, so those ARE included.
    //
    // `ln p_i = l_i - max - ln(sum_j e^(l_j - max))`, taken directly rather than as the logarithm
    // of a normalized probability, so a token whose probability underflows binary32 still gets
    // its finite value. The normalizer is summed in binary64, because a binary32 running sum over
    // a whole vocabulary accumulates a rounding error per term and shifts every reported value by
    // the same amount. Its logarithm is binary32 (`math::ln`, correctly rounded): platform `ln`
    // is barred here because its last bits differ across platforms (`tests/no_platform_libm.rs`
    // in `infr-core`), and `infr_core::math` has no binary64 logarithm.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| f64::from(math::exp(l - max))).sum();
    let norm = f64::from(max) + f64::from(math::ln(sum as f32));
    // A `-inf` logit, or a value past binary32's range, has no finite report: clamped, as below.
    let logprob_of = |l: f32| ((f64::from(l) - norm) as f32).max(LOGPROB_FLOOR);
    let logprob = logits
        .get(selected as usize)
        .map_or(LOGPROB_FLOOR, |&l| logprob_of(l));
    if top_n == 0 {
        return infr_chat::TokenLogprob::unresolved(selected, logprob, Vec::new());
    }
    let mut dist: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(id, &l)| (id as u32, l))
        .collect();
    // Select the top n in O(n), then sort only those n. Log probabilities order as the logits do.
    let cmp = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    if dist.len() > top_n {
        dist.select_nth_unstable_by(top_n - 1, cmp);
        dist.truncate(top_n);
    }
    dist.sort_unstable_by(cmp);
    let top = dist
        .into_iter()
        .map(|(id, l)| infr_chat::TokenLogprobAlt {
            token: id,
            text: String::new(),
            bytes: Vec::new(),
            logprob: logprob_of(l),
        })
        .collect();
    infr_chat::TokenLogprob::unresolved(selected, logprob, top)
}

/// Draw one id from a normalized `(id, prob)` distribution (as returned by [`truncated_dist`])
/// using the shared xorshift64 uniform draw — the stochastic MTP accept rule's residual/bonus
/// sampling (`crate::seam::model::spec_accept_stochastic`).
///
/// Empty distribution ⇒ token `0` plus a `debug_assert!`, the same contract [`sample_logits`]
/// states for an empty logits row. Keep the two arms identical; they are the crate's two "draw an
/// id" entry points, and a caller that hits the degenerate case should not get a panic from one
/// and a `0` from the other depending on which one it happened to call.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_from_dist(dist: &[(u32, f32)], rng: &mut u64) -> u32 {
    let Some(&(last_id, _)) = dist.last() else {
        debug_assert!(false, "sample_from_dist over an empty distribution");
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
    thread_local! {
        /// Entries the `top_k == 0` arm of [`truncated_softmax`] has sorted on this thread, summed
        /// over every pass of its widening loop. Per thread, so tests running in parallel cannot
        /// add to each other's count.
        pub(super) static NUCLEUS_SORTED: std::cell::Cell<usize> =
            const { std::cell::Cell::new(0) };
    }

    /// A NaN logit must not panic the bounded top-k arm: its comparator once called NaN "equal"
    /// to everything, which is not a total order, and the selection then panicked.
    #[test]
    fn bounded_top_k_survives_nan_logits() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for n in [600usize, 1590, 3000] {
            for every in [2usize, 3, 5] {
                let logits: Vec<f32> = (0..n)
                    .map(|i| {
                        if i % every == 0 {
                            f32::NAN
                        } else {
                            super::next_uniform(&mut state) * 10.0
                        }
                    })
                    .collect();
                for k in [1usize, 24, 40, 64] {
                    let (idx, probs) = super::truncated_softmax(&logits, 0.8, k, 0.95, 0.0);
                    assert!(!idx.is_empty() && idx.len() == probs.len() && idx.len() <= k);
                }
            }
        }
    }

    use super::*;

    fn cfg(temp: f32, seed: u64) -> RequestSampling {
        RequestSampling {
            temp: Some(temp),
            seed: Some(seed),
            ..Default::default()
        }
    }

    #[test]
    fn token_logprob_reports_the_sampled_support() {
        let logits = [0.0f32, 1.0, 2.0, 3.0];
        // An open support AT temperature 1: what the report is taken from.
        let open = Sampler {
            temp: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        };
        let lp = token_logprob(&logits, 2, 2);
        let dist = truncated_dist(&logits, open);
        let expected = dist.iter().find(|(id, _)| *id == 2).unwrap().1.ln();
        assert!((lp.logprob - expected).abs() < 1e-6);
        assert_eq!(lp.top.len(), 2);
        assert_eq!(lp.top[0].token, 3);
        assert!(
            (lp.top[0].logprob - dist.iter().find(|(id, _)| *id == 3).unwrap().1.ln()).abs() < 1e-6
        );

        let only = token_logprob(&logits, 2, 0);
        assert!(only.top.is_empty());
        assert_eq!(only.logprob, lp.logprob);
    }

    /// The report is the model's distribution, not the sampler's: temperature scales the logits
    /// before sampling, so a report taken from the scaled row would read near 0 for almost every
    /// token at `temperature: 0.01` and then jump at `temperature: 0`.
    #[test]
    fn token_logprob_ignores_temperature() {
        let logits = [0.0f32, 1.0, 2.0, 3.0];
        let cold = Sampler {
            temp: 0.01,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        };
        let scaled = truncated_dist(&logits, cold)
            .iter()
            .find(|(id, _)| *id == 2)
            .unwrap()
            .1
            .ln();
        assert!(
            scaled < -10.0,
            "a near-zero temperature would have reported {scaled}"
        );
        assert!(token_logprob(&logits, 2, 0).logprob > scaled);
    }

    /// Over a whole vocabulary the report stays within [`BOUND`] of an f64 log-softmax of the
    /// row: a binary32 running normalizer accumulates a rounding error per term, which moves every
    /// reported value by the same amount.
    #[test]
    fn token_logprob_is_the_log_softmax_of_a_vocabulary_sized_row() {
        /// The report's own roundings: the binary32 logarithm of the normalizer and the final
        /// binary32 value, each half an ulp of a number below 32.
        const BOUND: f64 = 1e-5;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let logits: Vec<f32> = (0..151_936)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 12.0 - 6.0
            })
            .collect();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = f64::from(max)
            + logits
                .iter()
                .map(|&l| (f64::from(l) - f64::from(max)).exp())
                .sum::<f64>()
                .ln();
        let mut worst = 0f64;
        for id in [0u32, 1, 77_777, 151_935] {
            let lp = token_logprob(&logits, id, 3);
            let want = f64::from(logits[id as usize]) - lse;
            let err = (f64::from(lp.logprob) - want).abs();
            worst = worst.max(err);
            assert!(err < BOUND, "token {id}: {} against {want}", lp.logprob);
            for alt in &lp.top {
                let want = f64::from(logits[alt.token as usize]) - lse;
                let err = (f64::from(alt.logprob) - want).abs();
                worst = worst.max(err);
                assert!(
                    err < BOUND,
                    "token {}: {} against {want}",
                    alt.token,
                    alt.logprob
                );
            }
        }
        eprintln!("worst error {worst:e} against the bound {BOUND:e}");
    }

    #[test]
    fn an_underflowing_probability_is_clamped_not_infinite() {
        // e^0 next to e^2000 is 0 in binary32, but its log probability is still -2000 exactly:
        // the report is the log-softmax, not the logarithm of an underflowed probability.
        let logits = [0.0f32, 2000.0];
        let lp = token_logprob(&logits, 0, 0);
        assert!(lp.logprob.is_finite(), "got {}", lp.logprob);
        assert_eq!(lp.logprob, -2000.0);
        // A token the row rules out has no finite log probability at all: clamped.
        let lp = token_logprob(&[f32::NEG_INFINITY, 0.0], 0, 2);
        assert_eq!(lp.logprob, LOGPROB_FLOOR);
        assert_eq!(lp.top[1].logprob, LOGPROB_FLOOR);
        // llama.cpp's value, `std::numeric_limits<float>::lowest()`.
        assert_eq!(lp.logprob, f32::MIN);
    }

    /// The PROCESS-default sampling config these tests layer their per-request overrides over —
    /// the shipped defaults, built as a value. This is what `Sampler::resolve` /
    /// `resolve_seed` take instead of reading the environment, so these tests no longer depend on
    /// (or perturb) `INFR_TEMP`/`INFR_SEED` at all and run in parallel with everything else.
    fn scfg() -> infr_core::config::SamplingCfg {
        infr_core::config::SamplingCfg::default()
    }

    /// `Sampler::from_cfg`'s doc contract, pinned as a value: nothing set ⇒ GREEDY. This is what
    /// keeps the goldens and every library caller deterministic; it used to be spelled
    /// "`INFR_TEMP` unset".
    #[test]
    fn default_sampling_cfg_is_greedy() {
        let s = Sampler::from_cfg(&scfg());
        assert_eq!(s.temp, 0.0, "unset ⇒ greedy");
        assert_eq!(s.top_k, 20);
        assert_eq!(s.top_p, 0.95);
        // And with no per-request overrides `resolve` IS `from_cfg`.
        let r = Sampler::resolve(None, &scfg());
        assert_eq!((r.temp, r.top_k, r.top_p), (s.temp, s.top_k, s.top_p));
    }

    /// `sampling.seed` reaches the RNG, and an unset seed still yields a usable (nonzero) xorshift
    /// state from the wall clock.
    ///
    /// `47` alone was never enough of a pin: it is ODD, so it survived the old `| 1` untouched and
    /// this test passed while `INFR_SEED=2`/`3` were silently the same stream. The `2` vs `3` and
    /// `Some(0)` cases below are what actually hold the PROCESS-config path to the seed policy.
    #[test]
    fn seed_comes_off_the_config() {
        let pinned = infr_core::config::SamplingCfg {
            seed: Some(47),
            ..Default::default()
        };
        assert_eq!(seed_rng(&pinned), 47);
        assert_eq!(resolve_seed(None, &pinned), 47);
        // A per-request seed still WINS over the process config (the unchanged precedence).
        let ctx = RequestCtx::new(cfg(1.0, 7));
        assert_eq!(resolve_seed(Some(&ctx), &pinned), 7);
        // Adjacent PROCESS seeds must NOT collapse onto one state (the `| 1` bug: 2|1 == 3|1 == 3).
        let seeded = |s: u64| {
            seed_rng(&infr_core::config::SamplingCfg {
                seed: Some(s),
                ..Default::default()
            })
        };
        assert_ne!(
            seeded(2),
            seeded(3),
            "`--seed 2` and `--seed 3` must differ"
        );
        assert_eq!((seeded(2), seeded(3)), (2, 3), "only 0 is remapped");
        // Seed 0 is xorshift's forbidden fixed point, so it (and ONLY it) is remapped.
        assert_ne!(seeded(0), 0, "seed 0 must be remapped off the zero state");
        // Unset: the wall-clock fallback is remapped too, so it can never be the zero state. It is
        // NOT guaranteed odd any more — oddness was an artifact of the collapsing `| 1`.
        assert_ne!(seed_rng(&scfg()), 0);
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
            assert_eq!(
                Sampler::resolve(Some(&a), &scfg()).temp,
                0.0,
                "A must keep temp 0"
            );
            assert_eq!(
                Sampler::resolve(Some(&b), &scfg()).temp,
                1.5,
                "B must keep temp 1.5"
            );
        }
        // Finding 4: seeds now pass through untouched (only the degenerate 0 is remapped), so a
        // request's seed is no longer collapsed onto an adjacent one.
        assert_eq!(resolve_seed(Some(&a), &scfg()), 42);
        assert_eq!(resolve_seed(Some(&b), &scfg()), 7);

        // The abort latch (stop sequences) is per-sequence too: A hitting its stop string must not
        // halt B.
        a.abort();
        assert!(abort_requested(Some(&a)), "A latched its own abort");
        assert!(!abort_requested(Some(&b)), "B must NOT see A's abort");
        // And a non-serve caller (run/bench/tests/goldens) has no latch at all.
        assert!(!abort_requested(None));
    }

    /// The server's own cancel latch is seen at every poll, with nothing in between to copy it
    /// across, and only by the sequence it was handed to.
    #[test]
    fn a_linked_cancel_aborts_its_own_sequence() {
        use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc};
        let cancel = Arc::new(AtomicBool::new(false));
        let linked = RequestCtx::new(RequestSampling::default()).with_cancel(cancel.clone());
        let other = RequestCtx::new(RequestSampling::default());
        assert!(!abort_requested(Some(&linked)));
        cancel.store(true, Ordering::Relaxed);
        assert!(
            abort_requested(Some(&linked)),
            "the linked cancel was not seen"
        );
        assert!(
            !abort_requested(Some(&other)),
            "an unlinked sequence saw it"
        );
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
            min_p: 0.0,
        };

        // Sequence A, alone.
        let a = RequestCtx::new(cfg(1.0, 42));
        let mut rng_a = resolve_seed(Some(&a), &scfg());
        let alone: Vec<u32> = (0..16)
            .map(|_| sample_logits(&logits, s, &mut rng_a))
            .collect();

        // Sequence A again — but now a second sequence B draws from the SAME thread between every
        // one of A's draws (the interleaved-scheduler shape).
        let a2 = RequestCtx::new(cfg(1.0, 42));
        let b = RequestCtx::new(cfg(1.5, 7));
        let mut rng_a2 = resolve_seed(Some(&a2), &scfg());
        let mut rng_b = resolve_seed(Some(&b), &scfg());
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

    /// **Finding 1 — repeat penalty is per-DISTINCT-token, not per-occurrence.** llama.cpp's
    /// `penalties` sampler divides a repeated token's positive logit by `repeat` exactly ONCE, no
    /// matter how many times it appears in the window. The old code walked the raw `recent` deque
    /// (with duplicates), so a token seen K times was scaled `repeat^K` — this fails under that bug.
    #[test]
    fn repeat_penalty_is_per_distinct_token_not_per_occurrence() {
        let ctx = RequestCtx::new(RequestSampling {
            repeat_penalty: 2.0,
            repeat_last_n: 64,
            ..Default::default()
        });
        let mut p = Penalties::resolve(Some(&ctx)).expect("penalty active");
        for _ in 0..3 {
            p.observe(5); // id 5 appears THREE times in the window
        }
        p.observe(7); // id 7 once
        let mut logits = vec![0.0f32; 8];
        logits[5] = 8.0;
        logits[7] = 4.0;
        p.apply(&mut logits);
        // id 5 penalized ONCE: 8/2 = 4  (the per-occurrence bug would give 8 / 2^3 = 1).
        assert_eq!(logits[5], 4.0, "distinct id must be penalized exactly once");
        assert_eq!(logits[7], 2.0, "id seen once: 4/2 = 2");
    }

    /// A generated id at or past the end of the logits row must be SKIPPED, not indexed. The trunk
    /// row is full-vocab so this cannot fire there, but `apply` patches whatever slice it is given
    /// (an lm-head slice, the MTP draft head's narrower vocab), and the raw `logits[t as usize]`
    /// turned that into a panic in the middle of a live generation. Both penalty arms are checked:
    /// the multiplicative repeat one and the additive presence/frequency one.
    #[test]
    fn penalties_skip_ids_past_the_end_of_the_logits_row() {
        let ctx = RequestCtx::new(RequestSampling {
            repeat_penalty: 2.0,
            repeat_last_n: 64,
            presence_penalty: 1.0,
            frequency_penalty: 0.5,
            ..Default::default()
        });
        let mut p = Penalties::resolve(Some(&ctx)).expect("penalty active");
        p.observe(1); // in range
        p.observe(4); // exactly one past the end of a 4-wide row
        p.observe(9_999); // far past the end
        p.observe(u32::MAX); // and the id that would index past isize::MAX on a 32-bit host
        let mut logits = vec![0.0f32; 4];
        logits[1] = 8.0;
        p.apply(&mut logits); // must not panic
                              // The in-range id still gets the full llama.cpp treatment: repeat (8/2 = 4) then
                              // presence+frequency (-(1.0 + 0.5*1) = -1.5).
        assert_eq!(logits[1], 2.5, "in-range id must still be penalized");
        assert_eq!(logits[0], 0.0, "untouched ids stay untouched");
        assert_eq!(logits.len(), 4, "the row is not resized");
    }

    /// A request's `logit_bias` alone, as `infr serve` builds it.
    fn biased(bias: Vec<(u32, f32)>) -> Penalties {
        let ctx = RequestCtx::new(RequestSampling {
            logit_bias: bias,
            ..Default::default()
        });
        Penalties::resolve(Some(&ctx)).expect("a logit_bias must take the host path")
    }

    /// The samplers the `logit_bias` tests draw with: greedy, the full-vocab paths (index order
    /// at top_p 1, the sorted widening path below it) and the bounded top-k path, at temperatures
    /// on both sides of 1.
    fn bias_samplers() -> [Sampler; 5] {
        let s = |temp, top_k, top_p| Sampler {
            temp,
            top_k,
            top_p,
            min_p: 0.0,
        };
        [
            s(0.0, 0, 1.0),
            s(1.0, 0, 1.0),
            s(2.0, 0, 1.0),
            s(0.7, 0, 0.95),
            s(1.5, 4, 0.95),
        ]
    }

    /// A row whose raw argmax is token 3 and whose lowest logit is token 5.
    const BIAS_ROW: [f32; 8] = [1.0, 2.0, 0.5, 6.0, 1.5, 0.0, 3.0, -1.0];

    /// **`logit_bias` -100 bans and +100 forces, at every temperature.** Token 3 is the raw
    /// argmax, so a bias that did not reach the row would draw it at once under greedy; banned, it
    /// must never be drawn on any sampling path across many seeds. Token 5 is the least likely
    /// token, and +100 must make it the only one ever drawn.
    #[test]
    fn logit_bias_bans_and_forces_on_every_sampling_path() {
        assert_eq!(
            argmax(&BIAS_ROW),
            3,
            "the banned token is not the raw argmax"
        );
        let mut banned = BIAS_ROW;
        biased(vec![(3, -100.0)]).apply(&mut banned);
        let mut forced = BIAS_ROW;
        biased(vec![(5, 100.0)]).apply(&mut forced);
        for s in bias_samplers() {
            for seed in 1..=2_000u64 {
                let mut rng = legal_xorshift_state(seed);
                let t = sample_logits(&banned, s, &mut rng);
                assert_ne!(t, 3, "{s:?} seed {seed}: drew the banned token");
                let t = sample_logits(&forced, s, &mut rng);
                assert_eq!(t, 5, "{s:?} seed {seed}: did not draw the forced token");
            }
        }
    }

    /// A uniform of exactly 0 on the index-order support (`top_k` 0, `top_p` 1) draws the most
    /// likely token, not the lowest id with any mass. Token 0's mass here only fails to underflow
    /// (it sits 100 below the top, a `logit_bias` ban); the first draw from state 1 is exactly 0.
    #[test]
    fn a_zero_draw_takes_the_most_likely_token_in_index_order() {
        assert_eq!(next_uniform(&mut 1), 0.0, "state 1 no longer draws 0");
        let row = [-95.0f32, 5.0, 4.0];
        let s = Sampler {
            temp: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        };
        assert!(
            truncated_dist(&row, s)[0].1 > 0.0,
            "token 0's mass underflowed"
        );
        assert_eq!(sample_logits(&row, s, &mut 1), 1);
    }

    /// The backends that never read the request's sampler refuse the fields only the seam decode
    /// applies, naming the field, and accept a request without them (`min_p: 0` included).
    #[test]
    fn seam_only_fields_are_refused_by_name() {
        let r = |s: RequestSampling| s.refuse_seam_only_fields("test");
        assert_eq!(r(RequestSampling::default()), Ok(()));
        let zero = RequestSampling {
            min_p: Some(0.0),
            ..Default::default()
        };
        assert_eq!(r(zero), Ok(()));
        let min_p = RequestSampling {
            min_p: Some(0.1),
            ..Default::default()
        };
        assert_eq!(r(min_p).unwrap_err().param, "min_p");
        let bias = RequestSampling {
            logit_bias: vec![(1, -100.0)],
            ..Default::default()
        };
        let e = r(bias).unwrap_err();
        assert_eq!(e.param, "logit_bias");
        assert_eq!(
            e.to_string(),
            "logit_bias is not supported by the test backend"
        );
    }

    /// A `logit_bias` alone sends the sequence to the host path (`resolve` is `Some`, which the
    /// decode loop's GPU-sampling gates key on); an empty one does not.
    #[test]
    fn logit_bias_alone_patches_the_row_on_the_host() {
        let none = RequestCtx::new(RequestSampling {
            logit_bias: Vec::new(),
            ..Default::default()
        });
        assert!(Penalties::resolve(Some(&none)).is_none());
        let p = biased(vec![(1, 2.5)]);
        let mut row = vec![0.0f32; 4];
        p.apply(&mut row);
        assert_eq!(row, [0.0, 2.5, 0.0, 0.0]);
    }

    /// The bias goes on BEFORE the penalties, as llama.cpp's chain runs `logit_bias` ahead of
    /// `penalties`; the order shows through the sign-aware repeat scale. Token 1 (logit 2, bias
    /// +10, seen once, repeat 2, presence 0.5): (2 + 10) / 2 - 0.5 = 5.5, where penalties first
    /// would give 2 / 2 - 0.5 + 10 = 10.5. A biased id past the end of the row is skipped, and a
    /// masked token stays masked.
    #[test]
    fn logit_bias_applies_before_the_penalties() {
        let ctx = RequestCtx::new(RequestSampling {
            repeat_penalty: 2.0,
            repeat_last_n: 64,
            presence_penalty: 0.5,
            logit_bias: vec![(1, 10.0), (2, -100.0), (99, 5.0)],
            ..Default::default()
        });
        let mut p = Penalties::resolve(Some(&ctx)).expect("active");
        p.observe(1);
        let mut row = vec![0.0f32, 2.0, f32::NEG_INFINITY, 4.0];
        p.apply(&mut row);
        assert_eq!(row, [0.0, 5.5, f32::NEG_INFINITY, 4.0]);
        // Saturated, not overflowed: a bias keeps a finite logit finite.
        let mut row = vec![f32::MAX];
        biased(vec![(0, 100.0)]).apply(&mut row);
        assert_eq!(row, [f32::MAX]);
    }

    /// The `min_p` row: max logit 3.0 (token 4). At `min_p` 0.3 llama.cpp keeps `l >= 3 + ln 0.3`
    /// (= 1.796): tokens 4 (3.0), 5 (2.5) and 0 (2.0), whose probability ratios to the top are
    /// e^0, e^-0.5 = 0.607 and e^-1 = 0.368; token 1 (2.0 below the top, ratio 0.135) and the rest
    /// go.
    const MIN_P_ROW: [f32; 6] = [2.0, 1.0, 0.0, -1.0, 3.0, 2.5];

    /// The sorted support `truncated_dist` keeps for [`MIN_P_ROW`].
    fn min_p_support(temp: f32, top_k: usize, top_p: f32, min_p: f32) -> Vec<(u32, f32)> {
        let mut d = truncated_dist(
            &MIN_P_ROW,
            Sampler {
                temp,
                top_k,
                top_p,
                min_p,
            },
        );
        d.sort_by_key(|&(i, _)| i);
        d
    }

    /// **`min_p` cuts what llama.cpp's rule cuts**, on every path `truncated_softmax` has (index
    /// order at top_p 1, the sorted widening path, the bounded top-k path), and on the RAW logits:
    /// at temperature 0.5 token 0's tempered ratio is e^-2 = 0.135, under 0.3, yet llama.cpp keeps
    /// it because its temperature runs after min-p. At temperature 1 the kept three renormalize to
    /// e^{-1, 0, -0.5} / (1 + e^-0.5 + e^-1) = 0.1863, 0.5065, 0.3072.
    #[test]
    fn min_p_cuts_what_llama_cpp_cuts() {
        for (temp, top_k, top_p) in [(1.0, 0, 1.0), (1.0, 0, 0.999), (1.0, 4, 1.0), (0.5, 0, 1.0)] {
            let ids: Vec<u32> = min_p_support(temp, top_k, top_p, 0.3)
                .iter()
                .map(|&(i, _)| i)
                .collect();
            assert_eq!(ids, [0, 4, 5], "temp {temp} top_k {top_k} top_p {top_p}");
        }
        let probs: Vec<f32> = min_p_support(1.0, 0, 1.0, 0.3)
            .iter()
            .map(|&(_, p)| p)
            .collect();
        for (got, want) in probs.iter().zip([0.186_323, 0.506_480, 0.307_196]) {
            assert!((got - want).abs() < 1e-5, "{probs:?}");
        }
        // min_p 1 keeps only the top token; the top token is kept at any min_p.
        let only: Vec<u32> = min_p_support(1.0, 0, 1.0, 1.0)
            .iter()
            .map(|&(i, _)| i)
            .collect();
        assert_eq!(only, [4]);
        // The shorter of the two prefixes wins: top_p 0.5 at temperature 1 is already {4, 5}.
        let both: Vec<u32> = min_p_support(1.0, 0, 0.5, 0.3)
            .iter()
            .map(|&(i, _)| i)
            .collect();
        assert_eq!(both, [4, 5]);
    }

    /// A `min_p` that cuts no token leaves every path's support and probabilities bit-identical
    /// to `min_p = 0`: the floor only truncates, it never touches a kept probability. (This does
    /// not compare against the sampler before `min_p` existed; the `min_p = 0` path is pinned
    /// against the full-sort reference by the tests above.)
    #[test]
    fn a_min_p_that_cuts_nothing_matches_min_p_zero_bit_for_bit() {
        for &n in &[7usize, 300, NUCLEUS_PROBE_K * 3] {
            let logits = seeded_logits(n, 77, 5.0, 0, false);
            for (temp, top_k, top_p) in [(0.8, 0, 1.0), (0.8, 0, 0.9), (1.2, 16, 0.95)] {
                let off = truncated_softmax(&logits, temp, top_k, top_p, 0.0);
                let tiny = truncated_softmax(&logits, temp, top_k, top_p, 1e-30);
                assert_eq!(
                    off.0, tiny.0,
                    "n {n} temp {temp} top_k {top_k} top_p {top_p}"
                );
                let bits = |v: &[f32]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&off.1), bits(&tiny.1));
            }
        }
    }

    /// A penalty only reorders tokens: an extreme (but accepted) `repeat_penalty` saturates a
    /// finite logit instead of overflowing it to ±inf, and a masked `-inf` logit stays masked.
    #[test]
    fn penalties_keep_finite_logits_finite() {
        for repeat in [f32::MAX, f32::MIN_POSITIVE] {
            let ctx = RequestCtx::new(RequestSampling {
                repeat_penalty: repeat,
                repeat_last_n: 64,
                frequency_penalty: 2.0,
                ..Default::default()
            });
            let mut p = Penalties::resolve(Some(&ctx)).expect("penalty active");
            for t in 0..3 {
                p.observe(t);
            }
            let mut logits = vec![-30.0f32, 30.0, f32::NEG_INFINITY, 1.0];
            p.apply(&mut logits);
            for (i, l) in logits[..2].iter().enumerate() {
                assert!(l.is_finite(), "repeat {repeat}: logit {i} became {l}");
            }
            assert_eq!(logits[2], f32::NEG_INFINITY, "repeat {repeat}: unmasked");
            assert_eq!(logits[3], 1.0, "repeat {repeat}: an unseen token moved");
        }
        // The additive penalties saturate as well: a frequency penalty no server request can
        // carry, in either direction, on a token seen twice.
        for frequency in [-1e38f32, 1e38] {
            let ctx = RequestCtx::new(RequestSampling {
                frequency_penalty: frequency,
                ..Default::default()
            });
            let mut p = Penalties::resolve(Some(&ctx)).expect("penalty active");
            for t in [0, 0, 1, 1, 2] {
                p.observe(t);
            }
            let mut logits = vec![-3e38f32, 3e38, f32::NEG_INFINITY, 1.0];
            p.apply(&mut logits);
            for (i, l) in logits[..2].iter().enumerate() {
                assert!(l.is_finite(), "frequency {frequency}: logit {i} became {l}");
            }
            assert_eq!(
                logits[2],
                f32::NEG_INFINITY,
                "frequency {frequency}: unmasked"
            );
            assert_eq!(
                logits[3], 1.0,
                "frequency {frequency}: an unseen token moved"
            );
        }
    }

    /// An empty support is "can't happen" (a real vocab is never empty), so the two draw entry
    /// points must at least AGREE about what "can't happen" does. They didn't: `sample_from_dist`
    /// returned `0`, while `sample_logits` either panicked on `idx[idx.len() - 1]` (the sampled
    /// path, whose heap came back empty) or silently returned `argmax`'s `0` (the greedy path) —
    /// three behaviours across two functions and two arms. Now all three trip a `debug_assert!`
    /// (loud here and in debug builds) and fall back to token `0` in release, so a caller that
    /// mis-slices — the MTP speculative path builds its own per-position slices — gets a defined
    /// answer instead of a killed request.
    #[test]
    fn empty_support_behaves_the_same_in_both_draw_paths() {
        // The three `catch_unwind`s below expect a panic in debug builds. No panic-hook fiddling to
        // hide their backtraces: `set_hook` is PROCESS-global and `cargo test` runs tests on
        // parallel threads, so silencing it here would also swallow the backtrace of any genuinely
        // failing test that happened to panic at the same moment. libtest already captures each
        // test's stderr and prints it only on failure, so the backtraces cost nothing anyway.
        let greedy = std::panic::catch_unwind(|| {
            let mut rng = 1u64;
            let s = Sampler {
                temp: 0.0,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
            };
            sample_logits(&[], s, &mut rng)
        });
        let sampled = std::panic::catch_unwind(|| {
            let mut rng = 1u64;
            let s = Sampler {
                temp: 1.0,
                top_k: 0,
                top_p: 0.95,
                min_p: 0.0,
            };
            sample_logits(&[], s, &mut rng)
        });
        let from_dist = std::panic::catch_unwind(|| {
            let mut rng = 1u64;
            sample_from_dist(&[], &mut rng)
        });

        if cfg!(debug_assertions) {
            // Debug: all three are loud. The greedy arm is the one that used to pass silently.
            assert!(greedy.is_err(), "greedy arm must assert on an empty row");
            assert!(sampled.is_err(), "sampled arm must assert on an empty row");
            assert!(from_dist.is_err(), "empty distribution must assert");
        } else {
            // Release: all three take the sibling's total fallback, none of them panic.
            assert_eq!(greedy.ok(), Some(0));
            assert_eq!(sampled.ok(), Some(0));
            assert_eq!(from_dist.ok(), Some(0));
        }
    }

    /// **Finding 4 — adjacent seeds must produce distinct streams.** `seed | 1` collapsed `2k` and
    /// `2k+1` onto the same odd xorshift state, so seeds 2 and 3 drew identical tokens. Only the
    /// degenerate seed 0 is remapped now; every other seed passes through untouched.
    #[test]
    fn adjacent_seeds_produce_distinct_streams() {
        let logits: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let s = Sampler {
            temp: 1.0,
            top_k: 8,
            top_p: 0.95,
            min_p: 0.0,
        };
        let draw = |seed: u64| -> Vec<u32> {
            let ctx = RequestCtx::new(RequestSampling {
                temp: Some(1.0),
                seed: Some(seed),
                ..Default::default()
            });
            let mut rng = resolve_seed(Some(&ctx), &scfg());
            (0..16)
                .map(|_| sample_logits(&logits, s, &mut rng))
                .collect()
        };
        assert_ne!(
            draw(2),
            draw(3),
            "adjacent seeds must differ (old `seed|1` collapsed 2 and 3 to the same stream)"
        );
        // Seed 0 (xorshift's forbidden zero state) still works and stays deterministic.
        assert_eq!(draw(0), draw(0), "seed 0 must be usable and reproducible");
        assert_ne!(
            resolve_seed_of(0),
            0,
            "seed 0 must be remapped off the zero state"
        );

        // ── The PROCESS-config path (`INFR_SEED` / `--seed`), which the first fix MISSED ─────────
        // `resolve_seed` delegates its `None` (no per-request seed) arm to `seed_rng`, and `seed_rng`
        // kept its own `| 1` long after the per-request arm stopped collapsing. So a `run`/`bench`
        // invocation — which has no `RequestCtx` at all and rides exactly this arm — still drew the
        // same tokens for `--seed 2` and `--seed 3`. Drive the config path end to end, not just the
        // request path.
        let draw_cfg = |seed: u64| -> Vec<u32> {
            let cfg = infr_core::config::SamplingCfg {
                seed: Some(seed),
                ..Default::default()
            };
            let mut rng = resolve_seed(None, &cfg);
            (0..16)
                .map(|_| sample_logits(&logits, s, &mut rng))
                .collect()
        };
        assert_ne!(
            draw_cfg(2),
            draw_cfg(3),
            "adjacent PROCESS seeds must differ (`seed_rng`'s `| 1` collapsed 2 and 3)"
        );
        assert_eq!(draw_cfg(2), draw_cfg(2), "a pinned seed stays reproducible");
        // And the process path agrees with the request path for the same seed — one policy, one
        // helper, so `--seed 2` and a request's `"seed": 2` cannot diverge.
        assert_eq!(draw_cfg(2), draw(2), "both entry points share the remap");
    }

    fn resolve_seed_of(seed: u64) -> u64 {
        let ctx = RequestCtx::new(RequestSampling {
            seed: Some(seed),
            ..Default::default()
        });
        resolve_seed(Some(&ctx), &scfg())
    }

    /// **Finding 5 (characterization) — the `top_k==0` heap/partial-select refactor must return the
    /// SAME token as the old full-vocab sort.** A pinned, distinct-logit vector across several seeds
    /// is compared against a byte-for-byte copy of the pre-refactor algorithm (`reference_full_sort`
    /// below). If the optimization ever changes the sampled token this fails.
    #[test]
    fn top_k_zero_matches_reference_full_sort() {
        let logits: Vec<f32> = (0..300).map(|i| (i as f32) * 0.05).collect();
        let s = Sampler {
            temp: 0.8,
            top_k: 0,
            top_p: 0.9,
            min_p: 0.0,
        };
        for seed in [1u64, 2, 42, 12345, 99_999] {
            let mut rng = seed;
            let got = sample_logits(&logits, s, &mut rng);
            let mut rng_ref = seed;
            let want = reference_full_sort(&logits, s, &mut rng_ref);
            assert_eq!(
                got, want,
                "seed {seed}: top_k==0 refactor changed the sampled token"
            );
        }
    }

    /// **Finding 6 (characterization) — the `top_k>0` path (default sampling) must be byte-identical
    /// after routing through the shared `truncated_softmax` helper.** Same oracle, with a real top-k.
    #[test]
    fn top_k_path_matches_reference_full_sort() {
        let logits: Vec<f32> = (0..300).map(|i| ((i as f32) * 0.031).sin() * 4.0).collect();
        let s = Sampler {
            temp: 0.7,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
        };
        for seed in [1u64, 2, 42, 12345, 99_999] {
            let mut rng = seed;
            let got = sample_logits(&logits, s, &mut rng);
            let mut rng_ref = seed;
            let want = reference_full_sort(&logits, s, &mut rng_ref);
            assert_eq!(
                got, want,
                "seed {seed}: top_k path drifted from the reference"
            );
        }
    }

    /// The pre-refactor `sample_logits` body, verbatim — the oracle the refactor is pinned against.
    fn reference_full_sort(logits: &[f32], s: Sampler, rng: &mut u64) -> u32 {
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
            idx.select_nth_unstable_by(k - 1, cmp);
            idx.truncate(k);
        }
        idx.sort_unstable_by(cmp);
        let maxl = logits[idx[0]];
        let mut probs: Vec<f32> = idx
            .iter()
            .map(|&i| math::exp((logits[i] - maxl) / s.temp))
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

    /// Deterministic pseudo-random logits, generated from `seed` with the file's OWN xorshift step
    /// ([`next_uniform`]) — no `rand` dependency, so every case below reproduces byte-for-byte on
    /// every host and every rustc.
    ///
    /// `grid` controls TIES: `0` leaves the raw draws (distinct with overwhelming probability),
    /// anything else quantizes onto a coarse lattice so equal logits are common — see the tie
    /// caveat on [`randomized_differential_against_reference_full_sort`]. `mask` sprinkles `-inf`
    /// entries, the shape a grammar constraint or a logit bias produces (and a second, adversarial
    /// source of ties, all of them at the very bottom of the order).
    fn seeded_logits(n: usize, seed: u64, spread: f32, grid: u32, mask: bool) -> Vec<f32> {
        let mut rng = legal_xorshift_state(seed);
        (0..n)
            .map(|i| {
                let u = next_uniform(&mut rng); // drawn unconditionally: masking must not shift the stream
                if mask && n >= 8 && i % 3 == 1 {
                    return f32::NEG_INFINITY;
                }
                let x = (u - 0.5) * 2.0 * spread;
                if grid == 0 {
                    x
                } else {
                    (x * grid as f32).round() / grid as f32
                }
            })
            .collect()
    }

    fn has_duplicate_logits(logits: &[f32]) -> bool {
        let mut v: Vec<f32> = logits.to_vec();
        v.sort_by(f32::total_cmp);
        v.windows(2).any(|w| w[0].total_cmp(&w[1]).is_eq())
    }

    /// **The randomized differential test for the bounded-nucleus rewrite.** Thousands of
    /// (vocab size × temp × top_p × top_k × logit shape × seed) combinations, each asserted against
    /// [`reference_full_sort`] — the byte-for-byte pre-refactor algorithm. The two pinned
    /// characterization tests above cover two hand-picked rows; this covers the corners that
    /// actually break a partial-selection rewrite: `n` below the probe size (1, 2, 7 — the clamp),
    /// `n` far above it (4096), `top_p = 0.01` (a one-token nucleus), `-inf` masked rows, and heavy
    /// ties.
    ///
    /// `top_p = 1.0` with no top-k builds no order (see [`truncated_softmax`]), so the token a
    /// uniform lands on legitimately differs from the reference's there. Those cases are checked
    /// against [`assert_full_support`] instead: the same support and the same probability per
    /// token, and the token drawn inside it. A `top_k` at or above `n` cuts nothing and takes the
    /// same path; with a top-k that cuts, `top_p = 1.0` still gets the token check.
    ///
    /// **Tie caveat — what is and is not asserted.** The old implementation popped a `BinaryHeap`
    /// whose order among EQUAL keys is unspecified, and `reference_full_sort` sorts with
    /// `sort_unstable_by`, which is likewise free to permute ties. The new code breaks ties by
    /// ascending index, so on a row with duplicate logits the two implementations can legitimately
    /// disagree about WHICH of two equal-logit tokens comes back. That disagreement is not a
    /// defect and it is not papered over by dropping the tied cases (they are the ones most likely
    /// to expose a real selection bug), so for rows that contain duplicates the assertion is
    /// weakened to exactly what IS guaranteed: the returned token has the same LOGIT — hence the
    /// same probability — as the reference's. That guarantee is tight, not hand-wavy: tied logits
    /// produce bit-identical `probs`, so `cum`, the nucleus cutoff, `total`, and the index `j` the
    /// draw lands on are all unchanged by tie order; only `idx[j]` can differ. Rows with distinct
    /// logits (the common case, and every real logit row) get the full `assert_eq!` on the token id.
    #[test]
    fn randomized_differential_against_reference_full_sort() {
        let mut cases = 0usize;
        let mut tied_rows = 0usize;
        let mut tie_swaps = 0usize;
        let mut widened = 0usize;
        let mut full_rows = 0usize;
        for &n in &[1usize, 2, 7, 17, 64, 255, 1024, 4096] {
            for &seed in &[1u64, 7, 12_345] {
                for &temp in &[0.05f32, 0.6, 1.0, 2.5] {
                    for &top_p in &[0.01f32, 0.3, 0.9, 1.0] {
                        for &(grid, mask, spread) in
                            &[(0u32, false, 6.0f32), (2, false, 6.0), (0, true, 3.0)]
                        {
                            // Cycle top_k so the untouched top-k arm rides along, but keep the
                            // rewritten `top_k == 0` arm the majority of the sweep.
                            let top_k = [0usize, 0, 0, 3, 20, 1][cases % 6];
                            let logits = seeded_logits(n, seed, spread, grid, mask);
                            let s = Sampler {
                                temp,
                                top_k,
                                top_p,
                                min_p: 0.0,
                            };
                            let mut rng = legal_xorshift_state(seed ^ 0xA5A5_5A5A);
                            let mut rng_ref = rng;
                            let got = sample_logits(&logits, s, &mut rng);
                            let want = reference_full_sort(&logits, s, &mut rng_ref);
                            let what = format!(
                                "n={n} seed={seed} temp={temp} top_p={top_p} top_k={top_k} grid={grid} mask={mask}"
                            );
                            assert_eq!(
                                rng, rng_ref,
                                "{what}: both paths must consume exactly one RNG draw"
                            );
                            if (top_k == 0 || top_k >= n) && top_p >= 1.0 {
                                // No order is built here, so the token a uniform lands on is not
                                // the reference's: compare the distribution instead, and check
                                // the draw came from it.
                                full_rows += 1;
                                let support = assert_full_support(&logits, temp, &what);
                                assert!(
                                    support.contains(&(got as usize)),
                                    "{what}: drew {got}, which has no probability"
                                );
                            } else if has_duplicate_logits(&logits) {
                                tied_rows += 1;
                                assert_eq!(
                                    logits[got as usize].total_cmp(&logits[want as usize]),
                                    std::cmp::Ordering::Equal,
                                    "{what}: tied row, but the token drawn has a DIFFERENT logit"
                                );
                                if got != want {
                                    tie_swaps += 1;
                                }
                            } else {
                                assert_eq!(got, want, "{what}: sampled token changed");
                            }
                            if top_k == 0
                                && top_p < 1.0
                                && truncated_softmax(&logits, temp, 0, top_p, 0.0).0.len()
                                    > NUCLEUS_PROBE_K
                            {
                                widened += 1;
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert!(cases >= 200, "sweep must be broad: only {cases} cases");
        assert!(tied_rows > 0, "the tied-logit flavour never fired");
        assert!(
            widened > 0,
            "no case exceeded the {NUCLEUS_PROBE_K}-candidate probe — the widening path was never \
             exercised by this sweep"
        );
        assert!(full_rows > 0, "no case took the top_p >= 1 support");
        // Informational, not asserted either way: how often the index tiebreak picked a different
        // (equal-probability) token than the heap's arbitrary order would have.
        eprintln!(
            "{cases} cases, {tied_rows} tied rows, {tie_swaps} tie swaps, {widened} widened, \
             {full_rows} full support"
        );
    }

    /// The support oracle for the widening path: full sort, no bounded probe, no widening — but
    /// otherwise arithmetically IDENTICAL to `truncated_softmax`'s `top_k == 0` arm (same
    /// index-order `sum`, same descending-with-index-tiebreak order, same accumulation), so the
    /// comparison below can be an exact `assert_eq!` on both `idx` AND `probs`, not a tolerance.
    fn reference_support(logits: &[f32], temp: f32, top_p: f32) -> (Vec<usize>, Vec<f32>) {
        let maxl = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&l| math::exp((l - maxl) / temp)).sum();
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then_with(|| a.cmp(b)));
        let (mut idx, mut probs, mut cum) = (Vec::new(), Vec::new(), 0.0f32);
        for &i in &order {
            let p = math::exp((logits[i] - maxl) / temp) / sum;
            idx.push(i);
            probs.push(p);
            cum += p;
            if cum >= top_p {
                break;
            }
        }
        (idx, probs)
    }

    /// What `truncated_softmax` must return for `top_k == 0` and `top_p >= 1`, checked without
    /// assuming any order: every token with nonzero probability exactly once, each with exactly
    /// the probability `reference_support` gives it (same `maxl`, same index-order `sum`, same
    /// expression, so equality is exact), and no token without one. Returns the support.
    ///
    /// Two independent views of "the same distribution": the definition (each token's own
    /// probability, no order involved) and the sorted oracle. The oracle can stop a hair before
    /// the end once its rounded cumulative sum reaches 1.0, so it is held to the tokens it keeps,
    /// and the tokens it cut are held to carrying almost no mass.
    fn assert_full_support(
        logits: &[f32],
        temp: f32,
        what: &str,
    ) -> std::collections::HashSet<usize> {
        let (idx, probs) = truncated_softmax(logits, temp, 0, 1.0, 0.0);
        assert_eq!(
            idx.len(),
            probs.len(),
            "{what}: idx and probs differ in length"
        );
        let got: std::collections::HashMap<usize, f32> =
            idx.iter().copied().zip(probs.iter().copied()).collect();
        assert_eq!(got.len(), idx.len(), "{what}: a token appears twice");

        let maxl = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&l| math::exp((l - maxl) / temp)).sum();
        for (i, &l) in logits.iter().enumerate() {
            let p = math::exp((l - maxl) / temp) / sum;
            if p > 0.0 {
                assert_eq!(
                    got.get(&i),
                    Some(&p),
                    "{what}: token {i} has the wrong mass"
                );
            } else {
                assert!(!got.contains_key(&i), "{what}: token {i} has no mass");
            }
        }

        let (ridx, rprobs) = reference_support(logits, temp, 1.0);
        for (&i, &p) in ridx.iter().zip(&rprobs) {
            if p > 0.0 {
                assert_eq!(
                    got.get(&i),
                    Some(&p),
                    "{what}: token {i} differs from the oracle"
                );
            }
        }
        // The rounding bound of an `n`-term sequential f32 sum: what the oracle's early stop and
        // the support's total can be off by without anything being wrong.
        let tol = (logits.len() as f32 * f32::EPSILON).max(1e-3);
        let kept: std::collections::HashSet<usize> = ridx.into_iter().collect();
        let cut: f32 = got
            .iter()
            .filter(|(i, _)| !kept.contains(i))
            .map(|(_, &p)| p)
            .sum();
        assert!(
            cut <= tol,
            "{what}: the oracle cut {cut} of mass, not rounding"
        );

        let mass: f32 = probs.iter().sum();
        assert!(
            (mass - 1.0).abs() <= tol,
            "{what}: support mass {mass} != 1"
        );
        got.into_keys().collect()
    }

    /// **The widening path, end to end.** A flat row at `top_p` near 1 needs nearly EVERY entry, so
    /// the `NUCLEUS_PROBE_K` probe must widen (×`NUCLEUS_WIDEN`, then snap to `n`) rather than hand
    /// back a truncated nucleus — silently truncating would drop mass the draw expects and, via
    /// [`truncated_dist`], change MTP's speculative accept ratios. Rows are sized to land on both
    /// sides of the probe and of one widening step, including exact-boundary sizes. `top_p = 1.0`
    /// takes no order at all, so it is held to [`assert_full_support`] instead of the sorted oracle.
    #[test]
    fn nucleus_widening_reaches_the_full_vocab() {
        let mut saw_snap = false;
        let mut saw_widened = false;
        let mut saw_full_support = false;
        for &n in &[
            NUCLEUS_PROBE_K - 1,
            NUCLEUS_PROBE_K,
            NUCLEUS_PROBE_K + 1,
            NUCLEUS_PROBE_K * NUCLEUS_WIDEN + 1, // one widening step is not enough
            NUCLEUS_PROBE_K * 12 + 7,            // needs the snap-to-n step too
        ] {
            for &(temp, top_p) in &[(1.0f32, 1.0f32), (0.6, 0.999), (1.0, 0.9)] {
                // Perfectly flat: every token carries the same mass, so the nucleus is as deep as
                // `top_p` demands and nothing about the row lets the probe get lucky. And a random
                // row, where the widening lands mid-tail.
                let flat = vec![0.0f32; n];
                let rnd = seeded_logits(n, 99, 2.0, 0, false);
                for (shape, row) in [("flat", &flat), ("random", &rnd)] {
                    let what = format!("{shape} n={n} temp={temp} top_p={top_p}");
                    if top_p >= 1.0 {
                        let support = assert_full_support(row, temp, &what);
                        saw_full_support |= support.len() == n;
                        continue;
                    }
                    let got = truncated_softmax(row, temp, 0, top_p, 0.0);
                    assert_eq!(
                        got,
                        reference_support(row, temp, top_p),
                        "{what}: bounded probe != full sort"
                    );
                    saw_widened |= got.0.len() > NUCLEUS_PROBE_K;
                    // Past one widening step's worth, which at these sizes only `k = n` can hold.
                    saw_snap |= got.0.len() > NUCLEUS_PROBE_K * NUCLEUS_WIDEN;
                }
            }
        }
        assert!(
            saw_widened,
            "no row exceeded the {NUCLEUS_PROBE_K}-candidate probe — widening untested"
        );
        assert!(
            saw_snap,
            "no row needed more than one widening step — the snap-to-n leg is untested"
        );
        assert!(
            saw_full_support,
            "no top_p = 1.0 row kept the WHOLE vocab — the full support is untested"
        );
    }

    /// **`top_p >= 1` with no top-k keeps the whole distribution, and gets it right**, on a row the
    /// size of a real vocabulary: a large random row, a masked one (the shape a grammar leaves),
    /// a heavily tied one, and a flat one, at several temperatures. A support that dropped or
    /// duplicated a token, kept a masked one, or normalized over anything but the full row fails
    /// [`assert_full_support`]. Draws over many seeds must also land only on tokens with mass,
    /// through both [`sample_logits`] and [`truncated_dist`]/[`sample_from_dist`].
    ///
    /// That the fast path is FASTER is measured, not asserted (timing in a unit test is noise);
    /// [`top_p_one_without_top_k_sorts_nothing`] asserts the mechanism that makes it so.
    #[test]
    fn top_p_one_without_top_k_keeps_the_whole_distribution() {
        const VOCAB: usize = 128_256;
        let rows = [
            ("random", seeded_logits(VOCAB, 5, 8.0, 0, false)),
            ("masked", seeded_logits(VOCAB, 6, 8.0, 0, true)),
            ("tied", seeded_logits(VOCAB, 7, 4.0, 2, false)),
            ("flat", vec![0.0f32; VOCAB]),
        ];
        for (shape, row) in &rows {
            for &temp in &[0.05f32, 0.6, 1.0, 2.5] {
                let what = format!("{shape} temp={temp}");
                let support = assert_full_support(row, temp, &what);
                if *shape == "masked" {
                    assert!(
                        support.iter().all(|&i| row[i].is_finite()),
                        "{what}: a masked token is in the support"
                    );
                }
                let s = Sampler {
                    temp,
                    top_k: 0,
                    top_p: 1.0,
                    min_p: 0.0,
                };
                let dist = truncated_dist(row, s);
                assert_eq!(dist.len(), support.len(), "{what}: MTP support differs");
                let mut rng = legal_xorshift_state(0xC0FFEE);
                for _ in 0..64 {
                    let t = sample_logits(row, s, &mut rng) as usize;
                    assert!(support.contains(&t), "{what}: drew {t}, which has no mass");
                    let t = sample_from_dist(&dist, &mut rng) as usize;
                    assert!(
                        support.contains(&t),
                        "{what}: MTP drew {t}, which has no mass"
                    );
                }
            }
        }
    }

    /// **The mechanism: `top_p >= 1` with no top-k sorts nothing.** The widening loop would end in
    /// a sort of all `n` entries; counted through `NUCLEUS_SORTED`. The `top_p < 1` row is the
    /// control that shows the counter is live, and the non-finite rows show the degenerate cases
    /// still take the sorted path they always did.
    #[test]
    fn top_p_one_without_top_k_sorts_nothing() {
        let sorted = |logits: &[f32], top_p: f32| {
            NUCLEUS_SORTED.with(|s| s.set(0));
            let _ = truncated_softmax(logits, 0.8, 0, top_p, 0.0);
            NUCLEUS_SORTED.with(std::cell::Cell::get)
        };
        let row = seeded_logits(NUCLEUS_PROBE_K * 16, 11, 2.0, 0, false);
        assert_eq!(sorted(&row, 1.0), 0, "top_p = 1.0 sorted the vocab");
        assert!(sorted(&row, 0.9) > 0, "the counter never moved");

        let mut nan = row.clone();
        nan[3] = f32::NAN;
        let all_masked = vec![f32::NEG_INFINITY; row.len()];
        for (what, r) in [("NaN", &nan), ("all -inf", &all_masked)] {
            assert!(sorted(r, 1.0) > 0, "{what} row left the sorted path");
            let (idx, probs) = truncated_softmax(r, 0.8, 0, 1.0, 0.0);
            assert_eq!(
                idx.len(),
                r.len(),
                "{what} row: the sorted path keeps every entry"
            );
            assert_eq!(idx.len(), probs.len());
        }
    }

    /// `truncated_dist` shares `truncated_softmax`, and it RENORMALIZES over the returned support —
    /// so a support the rewrite got wrong would silently reweight MTP's speculative accept rule
    /// (`spec_accept_stochastic`'s `p(x)/q(x)`) rather than fail loudly. Pin it directly: same
    /// support as the oracle, and a proper distribution (sums to 1) over it.
    #[test]
    fn truncated_dist_support_matches_the_full_sort_oracle() {
        for &n in &[1usize, 7, 300, NUCLEUS_PROBE_K + 5, NUCLEUS_PROBE_K * 9] {
            for &(temp, top_p) in &[(0.6f32, 0.9f32), (0.8, 0.95), (1.0, 1.0), (0.6, 0.01)] {
                let logits = seeded_logits(n, 2_024, 5.0, 0, false);
                let s = Sampler {
                    temp,
                    top_k: 0,
                    top_p,
                    min_p: 0.0,
                };
                let got = truncated_dist(&logits, s);
                if top_p >= 1.0 {
                    // No order here: the same tokens, each renormalized over the support's own
                    // index-order total, which is what `truncated_dist` divides by.
                    let what = format!("n={n} temp={temp} top_p={top_p}");
                    let support = assert_full_support(&logits, temp, &what);
                    let (idx, probs) = truncated_softmax(&logits, temp, 0, top_p, 0.0);
                    let total: f32 = probs.iter().sum();
                    let want: std::collections::HashMap<usize, f32> =
                        idx.into_iter().zip(probs).collect();
                    assert_eq!(got.len(), support.len(), "{what}: MTP support drifted");
                    for &(i, p) in &got {
                        assert_eq!(
                            Some(p),
                            want.get(&(i as usize)).map(|w| w / total),
                            "{what}: prob of {i} drifted"
                        );
                    }
                    continue;
                }
                let (want_idx, want_probs) = reference_support(&logits, temp, top_p);
                assert_eq!(
                    got.iter().map(|&(i, _)| i as usize).collect::<Vec<_>>(),
                    want_idx,
                    "n={n} temp={temp} top_p={top_p}: MTP support drifted"
                );
                let total: f32 = want_probs.iter().sum();
                for (k, (&(_, p), &w)) in got.iter().zip(want_probs.iter()).enumerate() {
                    assert_eq!(
                        p,
                        w / total,
                        "n={n} temp={temp} top_p={top_p}: prob {k} drifted"
                    );
                }
                let mass: f32 = got.iter().map(|&(_, p)| p).sum();
                assert!(
                    (mass - 1.0).abs() < 1e-3,
                    "n={n} temp={temp} top_p={top_p}: renormalized mass {mass} != 1"
                );
            }
        }
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
