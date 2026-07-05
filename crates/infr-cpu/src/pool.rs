//! Persistent spin-pool for the CPU op interpreter (threadpool restructure phase 2).
//!
//! Rayon's fork-join costs a wake/steal/sleep cycle per parallel op; between the ~400 ops of one
//! DiffusionGemma denoise graph that latency (and the deque/epoch plumbing measured at ~6% of
//! thread-time) is pure overhead. This pool keeps `N-1` workers alive across the whole graph:
//! a job is handed off by bumping a generation counter the workers spin on (~1µs handoff while
//! hot), tasks are claimed dynamically with a `fetch_add` cursor (straggler-proof), and workers
//! park after a short spin budget so an idle pool costs nothing (the host self-conditioning gap
//! between denoise steps, rayon-side MoeFfn work, and plain idle all put them to sleep).
//!
//! Scheduling only — every converted call site runs the exact same per-row math in the same
//! order as its rayon predecessor, so outputs are bit-identical.
//!
//! `INFR_CPU_NO_SPINPOOL=1` routes `run` through rayon instead (A/B + escape hatch).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

/// One in-flight job: an index-space of `n_tasks`, claimed dynamically by all threads.
type Job = &'static (dyn Fn(usize) + Sync);

struct Shared {
    /// The current job's closure, valid from the `seq` bump until every worker has checked in
    /// (`done == workers`) — `run` does not return (and the borrowed closure cannot die) before
    /// that, so a worker can never observe a stale/torn slot: it only reads `job` after seeing a
    /// NEW `seq`, and no new `seq` can be published while any worker is still on the old job.
    job: UnsafeCell<Option<Job>>,
    /// Generation counter: bumped once per job; workers spin on it changing.
    seq: AtomicUsize,
    /// Dynamic task cursor (`fetch_add` claim), reset per job.
    cursor: AtomicUsize,
    n_tasks: AtomicUsize,
    /// Workers that finished the current job (drained the cursor).
    done: AtomicUsize,
    /// A task panicked (caught per-task so `done` still advances; `run` re-panics).
    panicked: AtomicBool,
    shutdown: AtomicBool,
    /// [`SpinPool::pause`]: waiting workers park IMMEDIATELY (skip the spin budget). Set by call
    /// sites about to run a long rayon section (MoeFfn), cleared by the next `run`.
    pause: AtomicBool,
    /// Current ceiling for the adaptive spin budget — see [`SpinPool::set_budget_cap`].
    budget_cap: AtomicU32,
    /// Per-worker "I am parked" flags — see the park handshake in `worker_loop`.
    sleeping: Vec<AtomicBool>,
}

// SAFETY: `job` is only written by `run` while no worker is between jobs' check-ins (see the
// field doc), and only read by workers after an Acquire load of `seq` that the write
// happens-before (SeqCst bump).
unsafe impl Sync for Shared {}

pub(crate) struct SpinPool {
    shared: Arc<Shared>,
    handles: Vec<std::thread::JoinHandle<()>>,
    /// Worker thread count (callers participate too, so parallelism = workers + 1).
    workers: usize,
    /// `INFR_CPU_NO_SPINPOOL=1`: run jobs through rayon instead.
    rayon_fallback: bool,
    /// Serializes `run` — the pool holds ONE job; concurrent dispatch is a caller bug
    /// (converted call sites are all reached from the single-threaded `execute` op loop).
    in_run: AtomicBool,
}

/// CEILING for the adaptive per-worker spin budget (see `worker_loop`'s adaptive-budget doc).
/// Two measured failure modes bound the budget: too LONG and the spinning workers' SMT siblings
/// throttle the op loop's SERIAL bookkeeping between pool ops (per-op profile: RmsNorm/Add more
/// than DOUBLED; DG exec 2.87 → 3.18s at a fixed 32k) — rayon's MoeFfn itself was unharmed once
/// [`SpinPool::pause`] parked waiters for it. Too SHORT and dense prefill pays a worker wake per
/// op (qwen3 pp512 404 → 356 t/s at a fixed 1k). The adaptive budget collapses after a park and
/// regrows on jobs arriving mid-spin, so the ceiling can be generous. `INFR_CPU_SPIN` overrides.
static SPIN_LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();

fn spin_limit() -> u32 {
    *SPIN_LIMIT.get_or_init(|| {
        std::env::var("INFR_CPU_SPIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1 << 15)
    })
}

fn worker_loop(me: usize, shared: Arc<Shared>) {
    // Baseline generation is the CONSTRUCTION-time value (0), not a fresh load: a worker whose
    // OS thread starts late — after the first `run()` already bumped `seq` — must still join
    // that in-flight job (its check-in is what `run` is blocked on). Loading `seq` here instead
    // would make the late worker treat the live job as already-seen and deadlock the caller.
    let mut seen = 0usize;
    // ADAPTIVE spin budget, per worker: spinning is only worth its SMT-sibling tax (it throttles
    // the op loop's serial sections running on the paired hyperthread) when the next job arrives
    // before the budget runs out. Jobs arriving mid-spin → inter-op gaps are short (dense
    // prefill) → grow toward `spin_limit()`. Having to park → gaps are long (decode's serial
    // stretches, MoE's rayon section) → collapse to a near-immediate park. Measured: fixed
    // budgets force a 3-way tradeoff (qwen3 pp 416 vs tg 44 vs DG 2.87s — each best at a
    // DIFFERENT value); the gap history picks the right regime per phase automatically.
    const MIN_SPIN: u32 = 256;
    let mut budget = MIN_SPIN;
    loop {
        // ── Wait for a new generation ────────────────────────────────────────────────
        let mut spins = 0u32;
        let mut parked = false;
        loop {
            let s = shared.seq.load(Ordering::Acquire);
            if s != seen {
                seen = s;
                break;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            // `pause` skips the remaining spin budget — the dispatcher is telling us the cores
            // are about to be owned by a rayon section (MoeFfn).
            if spins < budget && !shared.pause.load(Ordering::Relaxed) {
                std::hint::spin_loop();
            } else {
                // Park handshake: publish the flag, RE-CHECK seq/shutdown (SeqCst on both sides
                // orders flag-publish vs. the dispatcher's seq-bump-then-read-flag), then park.
                // A wake that slips in between the re-check and `park()` is absorbed by park's
                // token semantics (`unpark` before `park` makes `park` return immediately).
                shared.sleeping[me].store(true, Ordering::SeqCst);
                if shared.seq.load(Ordering::SeqCst) != seen
                    || shared.shutdown.load(Ordering::SeqCst)
                {
                    shared.sleeping[me].store(false, Ordering::SeqCst);
                    continue;
                }
                std::thread::park();
                shared.sleeping[me].store(false, Ordering::SeqCst);
                spins = 0;
                parked = true;
            }
        }
        budget = if parked {
            MIN_SPIN
        } else {
            (budget * 4).min(shared.budget_cap.load(Ordering::Relaxed))
        };
        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }
        // ── Drain the task cursor ────────────────────────────────────────────────────
        // SAFETY: `job` was written before the observed `seq` bump and stays alive until this
        // worker (and all others) increment `done` — see `Shared::job`'s doc.
        let job = unsafe { (*shared.job.get()).expect("spin-pool: seq bumped without a job") };
        let n = shared.n_tasks.load(Ordering::Acquire);
        loop {
            let t = shared.cursor.fetch_add(1, Ordering::Relaxed);
            if t >= n {
                break;
            }
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(t))).is_err() {
                shared.panicked.store(true, Ordering::Release);
            }
        }
        shared.done.fetch_add(1, Ordering::Release);
    }
}

impl SpinPool {
    /// Thread count follows rayon's (`RAYON_NUM_THREADS` / available parallelism) so `-t` pins
    /// both pools identically.
    pub(crate) fn new() -> Self {
        let n_threads = rayon::current_num_threads().max(1);
        let workers = n_threads - 1;
        let rayon_fallback = std::env::var("INFR_CPU_NO_SPINPOOL").is_ok_and(|v| v != "0");
        let shared = Arc::new(Shared {
            job: UnsafeCell::new(None),
            seq: AtomicUsize::new(0),
            cursor: AtomicUsize::new(0),
            n_tasks: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            pause: AtomicBool::new(false),
            budget_cap: AtomicU32::new(spin_limit()),
            sleeping: (0..workers).map(|_| AtomicBool::new(false)).collect(),
        });
        let handles = (0..workers)
            .map(|me| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("infr-spin-{me}"))
                    .spawn(move || worker_loop(me, sh))
                    .expect("spin-pool: spawn failed")
            })
            .collect();
        SpinPool {
            shared,
            handles,
            workers,
            rayon_fallback,
            in_run: AtomicBool::new(false),
        }
    }

    /// Run `f(0..n_tasks)` across the pool (caller participates). Dynamic task claim; returns
    /// once every task ran AND every worker checked in (the check-in is what makes the borrowed
    /// closure's lifetime sound — see `Shared::job`).
    pub(crate) fn run(&self, n_tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        if n_tasks == 0 {
            return;
        }
        // Single-task jobs and worker-less pools short-circuit: no handoff, no wake.
        if n_tasks == 1 || self.workers == 0 {
            for t in 0..n_tasks {
                f(t);
            }
            return;
        }
        // Busy pool (a second graph executing concurrently on this backend — e.g. parallel
        // serve sessions) or explicit fallback: route through rayon, which handles concurrent
        // callers natively. The pool itself holds ONE job at a time.
        if self.rayon_fallback
            || self
                .in_run
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
        {
            use rayon::prelude::*;
            (0..n_tasks).into_par_iter().for_each(|t| f(t));
            return;
        }
        let sh = &self.shared;
        sh.pause.store(false, Ordering::Relaxed);
        // SAFETY: lifetime erasure of `f` — sound because this function does not return until
        // every worker has incremented `done`, after which no worker touches the slot again.
        unsafe {
            *sh.job.get() = Some(std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f));
        }
        sh.cursor.store(0, Ordering::Relaxed);
        sh.n_tasks.store(n_tasks, Ordering::Release);
        sh.done.store(0, Ordering::Relaxed);
        sh.seq.fetch_add(1, Ordering::SeqCst);
        for (i, flag) in sh.sleeping.iter().enumerate() {
            if flag.load(Ordering::SeqCst) {
                self.handles[i].thread().unpark();
            }
        }
        // Participate.
        loop {
            let t = sh.cursor.fetch_add(1, Ordering::Relaxed);
            if t >= n_tasks {
                break;
            }
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(t))).is_err() {
                sh.panicked.store(true, Ordering::Release);
            }
        }
        // Wait for every worker's check-in (they may still be draining the cursor's tail).
        // Periodic yield: under thread oversubscription (tests spawning several pools, a busy
        // rayon pool alongside) a pure spin here can starve the very workers it waits on.
        let mut spins = 0u32;
        while sh.done.load(Ordering::Acquire) < self.workers {
            spins += 1;
            if spins < spin_limit() {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        unsafe {
            *sh.job.get() = None;
        }
        self.in_run.store(false, Ordering::Release);
        if sh.panicked.swap(false, Ordering::AcqRel) {
            panic!("spin-pool: a task panicked (caught per-task; state may be incomplete)");
        }
    }

    /// Set the ceiling the adaptive per-worker spin budget may grow to. The op interpreter
    /// calls this per graph: a graph containing a rayon section (MoeFfn) gets a near-zero cap —
    /// spinning around those sections measurably starves them (qwen3moe pp512 123 -> 97 t/s even
    /// WITH `pause`) — while an all-pool dense graph gets the full budget (parking between every
    /// op cost qwen3 pp512 404 -> 356 t/s).
    pub(crate) fn set_budget_cap(&self, cap: u32) {
        // An explicit `INFR_CPU_SPIN` wins over the per-graph heuristic (experiment override).
        let cap = if std::env::var_os("INFR_CPU_SPIN").is_some() {
            spin_limit()
        } else {
            cap
        };
        self.shared.budget_cap.store(cap, Ordering::Relaxed);
    }

    /// Tell waiting workers to park immediately instead of finishing their spin budget — call
    /// before a long rayon section (MoeFfn) so 31 spinning threads don't starve it at the
    /// handover. Cleared automatically by the next `run`. Purely a scheduling hint: workers
    /// mid-job are unaffected, and a `run` racing this simply wakes them again.
    pub(crate) fn pause(&self) {
        self.shared.pause.store(true, Ordering::Relaxed);
    }

    /// Chunk `data` into `chunk`-sized pieces and run `f(chunk_index, piece)` across the pool,
    /// `grain` consecutive chunks per claimed task (coarsening for huge chunk counts). The
    /// last piece may be shorter (`data.len() % chunk`). Bit-identity: pure scheduling — each
    /// piece is processed by exactly one thread with unchanged math.
    pub(crate) fn for_chunks_mut<T: Send>(
        &self,
        data: &mut [T],
        chunk: usize,
        grain: usize,
        f: &(dyn Fn(usize, &mut [T]) + Sync),
    ) {
        let len = data.len();
        if len == 0 {
            return;
        }
        let n_chunks = len.div_ceil(chunk);
        let grain = grain.max(1);
        let n_tasks = n_chunks.div_ceil(grain);
        let base = SendPtr(data.as_mut_ptr());
        self.run(n_tasks, &move |task| {
            let c0 = task * grain;
            let c1 = (c0 + grain).min(n_chunks);
            for c in c0..c1 {
                let start = c * chunk;
                let end = (start + chunk).min(len);
                // SAFETY: chunk ranges are disjoint across tasks and in-bounds by construction.
                let piece =
                    unsafe { std::slice::from_raw_parts_mut(base.get().add(start), end - start) };
                f(c, piece);
            }
        });
    }

    /// `(0..n).map(f).collect()` across the pool, order-preserving.
    pub(crate) fn collect<T: Send>(&self, n: usize, f: &(dyn Fn(usize) -> T + Sync)) -> Vec<T> {
        let mut out: Vec<std::mem::MaybeUninit<T>> = Vec::with_capacity(n);
        // SAFETY: every index 0..n is written exactly once below before assume-init.
        unsafe { out.set_len(n) };
        let base = SendPtr(out.as_mut_ptr());
        self.run(n, &move |i| {
            // SAFETY: each task writes only its own slot.
            unsafe { base.get().add(i).write(std::mem::MaybeUninit::new(f(i))) };
        });
        // SAFETY: all n slots initialized (run returns only after every task completed; a panic
        // in `f` propagates out of `run` before we get here).
        unsafe { std::mem::transmute::<Vec<std::mem::MaybeUninit<T>>, Vec<T>>(out) }
    }
}

/// Raw base pointer that may cross thread boundaries; safety is argued at each use site
/// (disjoint index ranges per task). Accessed via [`SendPtr::get`], NOT the field — edition-2021
/// closures capture individual FIELDS, and a captured bare `*mut T` loses these unsafe impls.
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.seq.fetch_add(1, Ordering::SeqCst);
        for h in &self.handles {
            h.thread().unpark();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn spin_pool_runs_every_task_once() {
        let pool = SpinPool::new();
        for &n in &[1usize, 2, 7, 64, 1000, 10007] {
            let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
            pool.run(n, &|t| {
                hits[t].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                "n={n}: some task ran zero or multiple times"
            );
        }
    }

    #[test]
    fn spin_pool_chunks_and_collect() {
        let pool = SpinPool::new();
        let mut v = vec![0u32; 1000];
        pool.for_chunks_mut(&mut v, 16, 3, &|c, piece| {
            for (i, x) in piece.iter_mut().enumerate() {
                *x = (c * 16 + i) as u32;
            }
        });
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as u32));
        let got = pool.collect(257, &|i| i * 2);
        assert!(got.iter().enumerate().all(|(i, &x)| x == i * 2));
    }

    #[test]
    fn spin_pool_reusable_across_many_jobs() {
        let pool = SpinPool::new();
        let acc = AtomicU64::new(0);
        for _ in 0..200 {
            pool.run(32, &|_| {
                acc.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(acc.load(Ordering::Relaxed), 200 * 32);
    }
}
