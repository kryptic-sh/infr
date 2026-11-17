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
//! Clearing `kernels.cpu.spinpool` routes `run` through rayon instead
//! (A/B + escape hatch).

use infr_core::config::CpuCfg;
use std::any::Any;
use std::cell::UnsafeCell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

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
    /// Workers that finished the current job (drained the cursor, or stopped at a panic).
    done: AtomicUsize,
    /// A task of the current job panicked and its payload is in `payload`. The caller reads this
    /// once per job, so a job that did not panic never touches the mutex.
    panicked: AtomicBool,
    /// The FIRST panic payload of the current job, which `run` resumes on its caller. Later
    /// panics of the same job are dropped (see [`Shared::record_panic`]).
    payload: Mutex<Option<Box<dyn Any + Send>>>,
    shutdown: AtomicBool,
    /// Ceiling for the adaptive spin budget (constant `SpinPool::spin_limit` since phase 3 removed
    /// the last rayon section from the interpreter; kept a field for future per-graph tuning).
    budget_cap: AtomicU32,
    /// Per-worker "I am parked" flags — see the park handshake in `worker_loop`.
    sleeping: Vec<AtomicBool>,
}

// SAFETY: `job` is only written by `run` while no worker is between jobs' check-ins (see the
// field doc), and only read by workers after an Acquire load of `seq` that the write
// happens-before (SeqCst bump).
unsafe impl Sync for Shared {}

impl Shared {
    /// Claim and run tasks of the current job until the cursor passes `n`: the loop every thread
    /// of a job runs, the caller included.
    fn drain(&self, n: usize, job: &(dyn Fn(usize) + Sync)) {
        loop {
            let t = self.cursor.fetch_add(1, Ordering::Relaxed);
            if t >= n {
                break;
            }
            job(t);
        }
    }

    /// Run [`Shared::drain`], and turn a panic out of it into the job's recorded outcome instead
    /// of an unwind: the thread that panicked keeps going, so a worker survives to serve the next
    /// job and the caller still reaches its wait for the workers.
    ///
    /// A panic also ends the job early. Pushing the cursor to `n` makes every thread's next claim
    /// its last, so tasks no thread has claimed yet are skipped; tasks already running finish. A
    /// task that panics usually fails the same way on every other input (a bad stride, a wrong
    /// length), and running the rest only to panic in them too would take the panic hook — and,
    /// with `RUST_BACKTRACE` set, a backtrace capture — once per task.
    fn drain_catching(&self, n: usize, job: &(dyn Fn(usize) + Sync)) {
        if let Err(p) = catch_unwind(AssertUnwindSafe(|| self.drain(n, job))) {
            self.cursor.fetch_max(n, Ordering::Relaxed);
            self.record_panic(p);
        }
    }

    /// Keep `p` as the job's panic unless an earlier one is already kept.
    fn record_panic(&self, p: Box<dyn Any + Send>) {
        // Poison cannot carry a broken invariant here: the slot is one `Option` assignment.
        let mut slot = self.payload.lock().unwrap_or_else(PoisonError::into_inner);
        let extra = match *slot {
            None => {
                *slot = Some(p);
                None
            }
            Some(_) => Some(p),
        };
        self.panicked.store(true, Ordering::Release);
        drop(slot);
        if let Some(extra) = extra {
            // A payload's destructor is arbitrary code and may itself panic; unwinding out of
            // here would end a worker before its check-in and hang the caller. Such a payload is
            // leaked rather than dropped.
            if let Err(again) = catch_unwind(AssertUnwindSafe(move || drop(extra))) {
                std::mem::forget(again);
            }
        }
    }
}

pub(crate) struct SpinPool {
    shared: Arc<Shared>,
    handles: Vec<std::thread::JoinHandle<()>>,
    /// The placement every worker took and the caller takes during [`SpinPool::place_caller`]:
    /// [`infr_plat::cpu::spread_placement`] for this pool's threads, read once on the thread that
    /// built the pool, when `kernels.cpu.core_spread` is set. An affinity change after that is
    /// not picked up by this pool.
    pub(crate) placement: Option<infr_plat::cpu::Placement>,
    /// Worker thread count (callers participate too, so parallelism = workers + 1).
    workers: usize,
    /// `!kernels.cpu.spinpool`: run jobs through rayon instead.
    rayon_fallback: bool,
    /// CEILING for the adaptive per-worker spin budget, and the caller-side wait loop's
    /// yield threshold — `kernels.cpu.spin`, read ONCE at construction.
    ///
    /// Two measured failure modes bound the budget: too LONG and the spinning workers' SMT
    /// siblings throttle the op loop's SERIAL bookkeeping between pool ops (per-op profile:
    /// RmsNorm/Add more than DOUBLED; DG exec 2.87 → 3.18s at a fixed 32k) — rayon's MoeFfn itself
    /// was unharmed once a pause mechanism parked waiters for it (both since removed with the rayon
    /// sections themselves — phase 3 staged MoeFfn onto this pool). Too SHORT and dense prefill pays
    /// a worker wake per op (qwen3 pp512 404 → 356 t/s at a fixed 1k). The adaptive budget collapses
    /// after a park and regrows on jobs arriving mid-spin, so the ceiling can be generous.
    ///
    /// A FIELD, not a `OnceLock` memo: the value comes off the backend's `Config`, so a per-call
    /// read would be a `getenv`-shaped cost on the wait loop and a memo would pin the first pool's
    /// value process-wide.
    spin_limit: u32,
    /// Serializes `run` — the pool holds ONE job; concurrent dispatch is a caller bug
    /// (converted call sites are all reached from the single-threaded `execute` op loop).
    in_run: AtomicBool,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn worker_loop(me: usize, shared: Arc<Shared>) {
    // Baseline generation is the CONSTRUCTION-time value (0), not a fresh load: a worker whose
    // OS thread starts late — after the first `run()` already bumped `seq` — must still join
    // that in-flight job (its check-in is what `run` is blocked on). Loading `seq` here instead
    // would make the late worker treat the live job as already-seen and deadlock the caller.
    let mut seen = 0usize;
    // ADAPTIVE spin budget, per worker: spinning is only worth its SMT-sibling tax (it throttles
    // the op loop's serial sections running on the paired hyperthread) when the next job arrives
    // before the budget runs out. Jobs arriving mid-spin → inter-op gaps are short (dense
    // prefill) → grow toward `SpinPool::spin_limit`. Having to park → gaps are long (decode's serial
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
            if spins < budget {
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
        let job = unsafe { *shared.job.get() };
        let n = shared.n_tasks.load(Ordering::Acquire);
        match job {
            Some(job) => shared.drain_catching(n, job),
            None => shared.record_panic(Box::new("spin-pool: seq bumped without a job")),
        }
        // Reached on every path, since a panic in a task was caught above: a worker that skipped
        // this check-in would leave `run` waiting on `done` forever.
        shared.done.fetch_add(1, Ordering::Release);
    }
}

struct CollectGuard<'a, T> {
    slots: *mut std::mem::MaybeUninit<T>,
    initialized: &'a [AtomicBool],
}

impl<T> Drop for CollectGuard<'_, T> {
    fn drop(&mut self) {
        for (i, initialized) in self.initialized.iter().enumerate() {
            if initialized.load(Ordering::Acquire) {
                // SAFETY: each true flag is published only after its corresponding slot has been
                // initialized, and `run` waits for all tasks before propagating a panic.
                unsafe { std::ptr::drop_in_place(self.slots.add(i).cast::<T>()) };
            }
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl SpinPool {
    /// Thread count follows rayon's (`RAYON_NUM_THREADS` / available parallelism) so `-t` pins
    /// both pools identically. Both knobs (`spin`, `spinpool`) are read out of `cfg` HERE and
    /// never again — see [`SpinPool::spin_limit`].
    pub(crate) fn new(cfg: &CpuCfg) -> Self {
        Self::with_threads(cfg, rayon::current_num_threads())
    }

    /// [`SpinPool::new`] with `n_threads` threads (the caller and `n_threads - 1` workers; at
    /// least one) instead of rayon's count, for a test that needs workers whatever the machine
    /// (miri reports one CPU).
    pub(crate) fn with_threads(cfg: &CpuCfg, n_threads: usize) -> Self {
        let n_threads = n_threads.max(1);
        let workers = n_threads - 1;
        let rayon_fallback = !cfg.spinpool;
        let spin_limit = cfg.spin;
        let shared = Arc::new(Shared {
            job: UnsafeCell::new(None),
            seq: AtomicUsize::new(0),
            cursor: AtomicUsize::new(0),
            n_tasks: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            payload: Mutex::new(None),
            shutdown: AtomicBool::new(false),
            budget_cap: AtomicU32::new(spin_limit),
            sleeping: (0..workers).map(|_| AtomicBool::new(false)).collect(),
        });
        // Built on the thread constructing the pool, which is the thread that will dispatch to it,
        // so the placement also respects that thread's own affinity; a worker's is the process's,
        // never narrower than that.
        let placement = cfg
            .core_spread
            .then(|| infr_plat::cpu::spread_placement(n_threads))
            .flatten();
        let handles = (0..workers)
            .map(|me| {
                let sh = shared.clone();
                let placement = placement.clone();
                std::thread::Builder::new()
                    .name(format!("infr-spin-{me}"))
                    .spawn(move || {
                        // The worker lives as long as the pool, so its placement is never
                        // cleared; the thread exits with it.
                        if let Some(p) = &placement {
                            apply_or_warn(p);
                        }
                        worker_loop(me, sh)
                    })
                    .expect("spin-pool: spawn failed")
            })
            .collect();
        SpinPool {
            shared,
            handles,
            placement,
            workers,
            rayon_fallback,
            spin_limit,
            in_run: AtomicBool::new(false),
        }
    }

    /// Threads a job runs on: the workers and the caller.
    pub(crate) fn threads(&self) -> usize {
        self.workers + 1
    }

    /// Give the calling thread the pool's placement until the returned guard drops, which clears
    /// it again; `None` when the pool has no placement or the thread already has one of its own
    /// (an outer call's, or someone else's: that one stays in force). Two system calls here and
    /// one at the drop: the placement itself was worked out when the pool was built.
    pub(crate) fn place_caller(&self) -> Option<CallerPlacement> {
        let p = self.placement.as_ref()?;
        // `then`, not `then_some`: a guard built eagerly and discarded would clear the thread's
        // existing placement in its drop.
        apply_or_warn(p).then(|| CallerPlacement {
            _not_send: std::marker::PhantomData,
        })
    }

    /// Run `f(0..n_tasks)` across the pool (caller participates). Dynamic task claim; returns
    /// once every task ran AND every worker checked in (the check-in is what makes the borrowed
    /// closure's lifetime sound — see `Shared::job`).
    ///
    /// # Panics
    /// With the first panic payload of any task, on whichever thread it ran, once every thread
    /// has stopped running tasks. Tasks not started when that panic was caught are skipped (see
    /// [`Shared::drain_catching`]), so the job's outputs are left partly written. The pool stays
    /// usable.
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
            (0..n_tasks).into_par_iter().for_each(f);
            return;
        }
        let sh = &self.shared;
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
        //
        // NOTHING from the `seq` bump above to the end of the wait below may unwind. Workers run
        // `f`, which borrows the caller's frame through the lifetime-erased `job` slot, so
        // unwinding out of `run` before every worker has checked in would free that frame under
        // them. The caller's own share is therefore caught exactly as a worker's is, and its
        // panic is resumed only after the wait.
        sh.drain_catching(n_tasks, f);
        // Wait for every worker's check-in (they may still be draining the cursor's tail).
        // Periodic yield: under thread oversubscription (tests spawning several pools, a busy
        // rayon pool alongside) a pure spin here can starve the very workers it waits on.
        let mut spins = 0u32;
        while sh.done.load(Ordering::Acquire) < self.workers {
            spins += 1;
            if spins < self.spin_limit {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        unsafe {
            *sh.job.get() = None;
        }
        // Consume `panicked` (and its payload) BEFORE releasing the job slot, and release it
        // either way.
        //
        // The flag is pool-GLOBAL (it lives on `Shared`, written by this caller and by every
        // `worker_loop`), while `in_run` is what makes the pool single-job. Releasing first opens a
        // window in which another caller's `compare_exchange` succeeds, its task panics and sets
        // the flag, and THIS caller's swap then consumes it: we would panic for their failure and
        // they would return normally with incomplete task state (backlog B25).
        //
        // Reading it while still holding `in_run` closes that window — no second job can have
        // started, so the flag can only be ours. The store must still happen before the unwind,
        // or an unwinding caller would leave `in_run` latched `true` and every later `run` would
        // silently take the rayon fallback for the life of the process.
        let payload = if sh.panicked.load(Ordering::Acquire) {
            sh.panicked.store(false, Ordering::Relaxed);
            sh.payload
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
        } else {
            None
        };
        self.in_run.store(false, Ordering::Release);
        if let Some(p) = payload {
            resume_unwind(p);
        }
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
        let initialized: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
        // ONE `as_mut_ptr` feeds both the tasks' writes and the unwind guard's drops. Calling it
        // twice would reborrow `out` mutably a second time, popping the first pointer's tag off
        // the borrow stack — so the guard, which runs only while unwinding and therefore only
        // AFTER the second pointer exists, would be dropping through a pointer the aliasing model
        // already considers dead. Same address either way, so nothing observable changes today;
        // it is UB by the model, not by the codegen.
        let base = SendPtr(out.as_mut_ptr());
        let guard = CollectGuard {
            slots: base.get(),
            initialized: &initialized,
        };
        self.run(n, &|i| {
            // SAFETY: each task writes only its own slot.
            unsafe { base.get().add(i).write(std::mem::MaybeUninit::new(f(i))) };
            initialized[i].store(true, Ordering::Release);
        });
        // Every slot is initialized after a successful `run`; disable the unwind-only destructor.
        std::mem::forget(guard);

        // The rebuild is done via `from_raw_parts` rather than
        // `transmute::<Vec<MaybeUninit<T>>, Vec<T>>`: `Vec` is not `repr(C)`, so transmuting
        // between two instantiations of it is not a language guarantee even though
        // `MaybeUninit<T>` and `T` have identical layout — the compiler is free to lay out
        // `Vec<A>` and `Vec<B>` differently. Taking the pointer/len/capacity apart and handing
        // them back to `Vec::from_raw_parts` is the sanctioned form and relies only on the
        // documented `MaybeUninit<T>`-to-`T` layout equivalence. (`Vec::into_raw_parts` is still
        // unstable, hence the manual `forget`.)
        let (ptr, len, cap) = (out.as_mut_ptr(), out.len(), out.capacity());
        std::mem::forget(out);
        // SAFETY: `ptr`/`len`/`cap` come straight from a `Vec<MaybeUninit<T>>` that we have
        // `forget`ten (so it will not also free the allocation), `MaybeUninit<T>` has the same
        // size and alignment as `T`, and all `len` elements are initialized per the argument
        // above — so the allocation is valid to reinterpret as a `Vec<T>` of the same capacity.
        unsafe { Vec::from_raw_parts(ptr as *mut T, len, cap) }
    }
}

/// Put the calling thread on the pool's placement, so the pool's threads stay off each other's
/// SMT siblings while the OS still places them. Left to the OS, Windows put the spin pool's
/// threads on SMT siblings of each other, which cost dense prefill a large share of its rate (see
/// the CHANGELOG). Returns whether the placement was applied (`false` when the thread already
/// had one). A failure only costs speed, so it is reported once per process and not again.
fn apply_or_warn(p: &infr_plat::cpu::Placement) -> bool {
    match infr_plat::cpu::apply_placement(p) {
        Ok(applied) => applied,
        Err(e) => {
            warn_placement_once(&e);
            false
        }
    }
}

fn warn_placement_once(e: &std::io::Error) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!("spin-pool: thread placement failed, left to the OS: {e}");
    }
}

/// The caller's placement from [`SpinPool::place_caller`]; dropping it clears the placement on
/// the thread that drops it, so it is not `Send`.
pub(crate) struct CallerPlacement {
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for CallerPlacement {
    fn drop(&mut self) {
        if let Err(e) = infr_plat::cpu::clear_placement() {
            warn_placement_once(&e);
        }
    }
}

/// Raw base pointer that may cross thread boundaries; safety is argued at each use site
/// (disjoint index ranges per task). Accessed via [`SendPtr::get`], NOT the field — edition-2021
/// closures capture individual FIELDS, and a captured bare `*mut T` loses these unsafe impls.
pub(crate) struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
// not instrumented: per-task pointer accessors in the dispatch hot loop, too small to probe
impl<T> SendPtr<T> {
    pub(crate) fn new(p: *mut T) -> Self {
        SendPtr(p)
    }
    pub(crate) fn get(&self) -> *mut T {
        self.0
    }
}
// not instrumented: trivial Copy-clone in hot loops; a span here would be pure overhead
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}

#[cfg_attr(infr_profile, infr_prof::instrument)]
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
        let pool = SpinPool::new(&CpuCfg::default());
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

    /// B25's observable consequence: after a panicking job the pool must still be a POOL.
    ///
    /// The race itself — a second caller slipping into the window between `in_run.store(false)` and
    /// the `panicked` swap — is not deterministically reproducible without pausing a thread
    /// mid-teardown, so this pins the property the reorder must not break instead: the panic
    /// propagates, `in_run` is released, and the next job neither inherits the flag nor gets
    /// silently demoted to the rayon fallback.
    #[test]
    fn a_panicking_job_leaves_the_pool_reusable_and_does_not_leak_its_flag() {
        let pool = SpinPool::new(&CpuCfg::default());
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(64, &|t| {
                if t == 17 {
                    panic!("task boom");
                }
            });
        }));
        std::panic::set_hook(prev);
        assert!(caught.is_err(), "a task panic must propagate to the caller");
        assert!(
            !pool.in_run.load(Ordering::Acquire),
            "the job slot must be released even when run() unwinds, or every later run() \
             silently falls back to rayon for the life of the process"
        );
        assert!(
            !pool.shared.panicked.load(Ordering::Acquire),
            "the panic flag must have been consumed, not left for the next job to inherit"
        );

        // The next job runs normally: it must not re-panic on a stale flag, and it must still be
        // the spin pool doing the work (every task exactly once).
        let hits: Vec<AtomicU64> = (0..256).map(|_| AtomicU64::new(0)).collect();
        pool.run(256, &|t| {
            hits[t].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn collect_drops_completed_values_when_another_task_panics() {
        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let pool = SpinPool::new(&CpuCfg::default());
        let drops = Arc::new(AtomicUsize::new(0));
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = pool.collect(2, &|i| {
                if i == 1 {
                    panic!("task boom");
                }
                DropCounter(Arc::clone(&drops))
            });
        }));
        std::panic::set_hook(prev);

        assert!(caught.is_err(), "the task panic must propagate");
        assert_eq!(
            drops.load(Ordering::Relaxed),
            1,
            "the value completed before the panic must be dropped"
        );
    }

    #[test]
    fn spin_pool_chunks_and_collect() {
        let pool = SpinPool::new(&CpuCfg::default());
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
        let pool = SpinPool::new(&CpuCfg::default());
        let acc = AtomicU64::new(0);
        for _ in 0..200 {
            pool.run(32, &|_| {
                acc.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(acc.load(Ordering::Relaxed), 200 * 32);
    }

    /// `kernels.cpu.spin` reaches the two places it is used — the workers' adaptive-budget ceiling
    /// and the caller-side wait loop's yield threshold — from the `CpuCfg` the pool was built with,
    /// and the SHIPPED default is still `1 << 15`. This used to be a process-wide `OnceLock`
    /// memo of the knob's env spelling, so a second pool could not see a different value at all;
    /// two pools with different budgets in one test is what could not be written before.
    #[test]
    fn spin_budget_comes_off_the_config_per_pool() {
        assert_eq!(CpuCfg::default().spin, 1 << 15);
        let dflt = SpinPool::new(&CpuCfg::default());
        assert_eq!(dflt.spin_limit, 1 << 15);
        assert_eq!(dflt.shared.budget_cap.load(Ordering::Relaxed), 1 << 15);

        let tight = SpinPool::new(&CpuCfg {
            spin: 4,
            ..CpuCfg::default()
        });
        assert_eq!(tight.spin_limit, 4);
        assert_eq!(tight.shared.budget_cap.load(Ordering::Relaxed), 4);
        // Scheduling-only knob: a near-zero budget parks workers immediately but must still run
        // every task exactly once.
        let hits: Vec<AtomicU64> = (0..777).map(|_| AtomicU64::new(0)).collect();
        tight.run(777, &|t| {
            hits[t].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    /// `kernels.cpu.spinpool` cleared routes `run` through rayon —
    /// scheduling only, so the observable contract is unchanged: every task still runs once.
    #[test]
    fn cleared_spinpool_falls_back_to_rayon() {
        assert!(CpuCfg::default().spinpool);
        assert!(!SpinPool::new(&CpuCfg::default()).rayon_fallback);

        let pool = SpinPool::new(&CpuCfg {
            spinpool: false,
            ..CpuCfg::default()
        });
        assert!(pool.rayon_fallback);
        let hits: Vec<AtomicU64> = (0..1000).map(|_| AtomicU64::new(0)).collect();
        pool.run(1000, &|t| {
            hits[t].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    // ── A panicking task ─────────────────────────────────────────────────────────────────────
    //
    // Each test runs its body through `within`, so a pool that hangs on a panic fails the test
    // instead of stalling the run, and builds its pool with `PANIC_THREADS` threads so it has
    // workers under miri too, which reports one CPU.

    use crate::test_util::{panic_of, within};

    const PANIC_THREADS: usize = 4;

    fn on_worker() -> bool {
        std::thread::current()
            .name()
            .is_some_and(|n| n.starts_with("infr-spin-"))
    }

    /// Spin, yielding, until `flag` is set.
    fn wait_for(flag: &AtomicBool) {
        while !flag.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
    }

    /// The pool is still a whole pool: the next job runs every task once, and every worker takes a
    /// share of it, so none of them died. Every thread holds its first task until each worker has
    /// one; otherwise a fast worker could drain the job before a slow one woke.
    fn assert_whole(pool: &SpinPool) {
        assert!(
            !pool.in_run.load(Ordering::Acquire),
            "the job slot was left taken"
        );
        let n = 8 * pool.threads();
        let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
        let workers_seen = Mutex::new(std::collections::HashSet::new());
        let all_seen = AtomicBool::new(false);
        pool.run(n, &|t| {
            if on_worker() {
                let mut seen = workers_seen.lock().unwrap();
                seen.insert(std::thread::current().id());
                if seen.len() == pool.workers {
                    all_seen.store(true, Ordering::Release);
                }
            }
            wait_for(&all_seen);
            hits[t].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    /// A task panicking on a WORKER reaches the caller as that panic, with its own message.
    #[test]
    fn a_worker_panic_reaches_the_caller_with_its_payload() {
        within(|| {
            let pool = SpinPool::with_threads(&CpuCfg::default(), PANIC_THREADS);
            let worker_failed = AtomicBool::new(false);
            let msg = panic_of(|| {
                pool.run(4 * PANIC_THREADS, &|t| {
                    if on_worker() {
                        worker_failed.store(true, Ordering::Release);
                        panic!("worker task {t} failed");
                    }
                    // The caller holds its first task until a worker has failed, so the failure
                    // is a worker's whatever the scheduling.
                    wait_for(&worker_failed);
                });
            });
            let msg = msg.expect("the worker's panic did not reach the caller");
            assert!(msg.starts_with("worker task "), "payload lost: {msg:?}");
            assert_whole(&pool);
        })
    }

    /// A panic in the CALLER's own share propagates only after every worker has finished the task
    /// it was running: those tasks borrow the caller's frame, which the unwind frees.
    #[test]
    fn a_panic_in_the_callers_share_waits_for_the_workers() {
        within(|| {
            let pool = SpinPool::with_threads(&CpuCfg::default(), PANIC_THREADS);
            let caller_failing = AtomicBool::new(false);
            let started = AtomicUsize::new(0);
            let finished = AtomicUsize::new(0);
            let msg = panic_of(|| {
                pool.run(4 * PANIC_THREADS, &|t| {
                    if on_worker() {
                        started.fetch_add(1, Ordering::SeqCst);
                        wait_for(&caller_failing);
                        // Still running well after the caller's task has panicked.
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        finished.fetch_add(1, Ordering::SeqCst);
                    } else {
                        caller_failing.store(true, Ordering::Release);
                        panic!("caller task {t} failed");
                    }
                });
            });
            let msg = msg.expect("the caller's panic was lost");
            assert!(msg.starts_with("caller task "), "payload lost: {msg:?}");
            assert_eq!(
                finished.load(Ordering::SeqCst),
                started.load(Ordering::SeqCst),
                "run() unwound while workers were still running tasks that borrow its frame"
            );
            assert_whole(&pool);
        })
    }

    /// When every task panics, a thread stops at its first one: tasks nobody has started are
    /// skipped, so a job that fails everywhere costs a panic per thread, not one per task.
    #[test]
    fn a_panic_skips_the_tasks_nobody_started() {
        within(|| {
            let pool = SpinPool::with_threads(&CpuCfg::default(), PANIC_THREADS);
            let n = if cfg!(miri) { 64 } else { 10_000 };
            let started = AtomicUsize::new(0);
            let msg = panic_of(|| {
                pool.run(n, &|t| {
                    started.fetch_add(1, Ordering::Relaxed);
                    panic!("task {t} failed");
                });
            });
            let msg = msg.expect("no panic reached the caller");
            assert!(msg.starts_with("task "), "payload lost: {msg:?}");
            let started = started.load(Ordering::Relaxed);
            assert!(
                started <= pool.threads(),
                "{started} of {n} tasks ran after the first panic on {} threads",
                pool.threads()
            );
            assert_whole(&pool);
        })
    }

    /// The pool serves job after job with failures among them, including a failure in a nested
    /// `run` (which goes through rayon while this pool is busy) inside one of its tasks.
    #[test]
    fn the_pool_keeps_serving_after_panics() {
        within(|| {
            let pool = SpinPool::with_threads(&CpuCfg::default(), PANIC_THREADS);
            for round in 0..if cfg!(miri) { 2 } else { 20 } {
                let msg = panic_of(|| {
                    pool.run(2 * PANIC_THREADS, &|t| {
                        if t == round % (2 * PANIC_THREADS) {
                            panic!("round {round} failed");
                        }
                    });
                });
                assert_eq!(
                    msg.as_deref(),
                    Some(format!("round {round} failed").as_str())
                );
                assert_whole(&pool);
            }
            let msg = panic_of(|| {
                pool.run(PANIC_THREADS, &|t| {
                    if t == 1 {
                        pool.run(8, &|u| {
                            if u == 5 {
                                panic!("nested task {u} failed");
                            }
                        });
                    }
                });
            });
            assert_eq!(msg.as_deref(), Some("nested task 5 failed"));
            assert_whole(&pool);
        })
    }
}
