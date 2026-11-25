//! The process-wide `reqwest` clients every hub request goes through, and the HF bearer token.
//!
//! THREE clients, each built once (a `Client` owns a connection pool + TLS config; building one
//! per request means a fresh TLS handshake per shard — 40 of them for a 40-shard model).
//!
//! The timeout split is deliberate and is the part a later reader is most likely to "fix" wrongly:
//!
//!   * `connect_timeout` on ALL of them — a server that never completes the TCP/TLS handshake must
//!     not hang `infr pull` (and `infr run`'s auto-pull) forever with no recovery but Ctrl-C, which
//!     is exactly what reqwest's default of NO timeout does.
//!   * a total `timeout` ONLY on the metadata/HEAD clients. `Client::timeout` bounds the WHOLE
//!     request including reading the body, so putting it on the download client would abort every
//!     multi-GB model download that legitimately takes longer than the limit. Metadata requests are
//!     a few KB, so a total bound is right there and wrong on the download path. A stalled *body*
//!     mid-download is not fatal here anyway: the partial is kept and the next run resumes it.

use infr_core::error::{Error, Result};
use reqwest::blocking::Client;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::{sync::OnceLock, time::Duration};

/// TCP+TLS handshake budget. Generous enough for a slow link, short enough to fail rather than hang.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Whole-request budget for the small (few-KB) metadata calls: the model API GET and the LFS HEAD.
const METADATA_TIMEOUT: Duration = Duration::from_secs(30);

/// The client used for BODY transfers (model blobs, companions). No total timeout — see above.
///
/// ONE client shared by every concurrent shard download on purpose: a `reqwest::blocking::Client`
/// is `Sync` and its connection pool is what lets N workers hold N sockets to the same host
/// without N TLS handshakes' worth of setup each time one finishes.
pub(crate) fn download_client() -> Result<&'static Client> {
    static CLIENT: OnceLock<std::result::Result<Client, String>> = OnceLock::new();
    shared(&CLIENT, "download", |b| b)
}

/// The client used for the HF model API (`/api/models/...`) and for [`crate::ranged::probe`]'s
/// `HEAD`. Both are small, redirect-following requests that must not hang, which is what separates
/// them from the download client; only [`head_client`]'s HEAD depends on seeing the 302 itself.
pub(crate) fn api_client() -> Result<&'static Client> {
    static CLIENT: OnceLock<std::result::Result<Client, String>> = OnceLock::new();
    shared(&CLIENT, "API", |b| b.timeout(METADATA_TIMEOUT))
}

/// The client used by `head_lfs_sha`. Redirects stay DISABLED — load-bearing: the `X-Linked-Etag`
/// sha256 is on huggingface.co's 302, not on the CDN's final 200, so following the redirect loses it.
pub(crate) fn head_client() -> Result<&'static Client> {
    static CLIENT: OnceLock<std::result::Result<Client, String>> = OnceLock::new();
    shared(&CLIENT, "HEAD", |b| {
        b.timeout(METADATA_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
    })
}

/// Build-once accessor for one of the shared clients. The build result (including a failure) is
/// cached, so a broken TLS backend reports the same error every time instead of being retried per
/// request; the error is returned, never panicked, so a pull can still fail gracefully.
fn shared(
    cell: &'static OnceLock<std::result::Result<Client, String>>,
    what: &str,
    extra: impl FnOnce(reqwest::blocking::ClientBuilder) -> reqwest::blocking::ClientBuilder,
) -> Result<&'static Client> {
    cell.get_or_init(|| {
        extra(
            Client::builder()
                .user_agent("infr-hub/0.1")
                .connect_timeout(CONNECT_TIMEOUT),
        )
        .build()
        .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| Error::Other(format!("building HTTP {what} client: {e}")))
}

/// The HuggingFace access token, for gated/private repos. `HF_TOKEN` is HuggingFace's own spelling,
/// shared with `huggingface_hub` and llama.cpp — not an `INFR_*` knob, so it is not configuration.
pub(crate) fn token() -> Option<String> {
    usable_token(std::env::var("HF_TOKEN").ok())
}

/// An unset and an EMPTY `HF_TOKEN` both mean "no token". Docker and CI env files routinely carry
/// `HF_TOKEN=`; sending that as `Authorization: Bearer ` can draw a 401 on a public repo. What is
/// sent is the trimmed value, since a trailing newline or space from such a file is not part of it.
fn usable_token(raw: Option<String>) -> Option<String> {
    raw.map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

/// The pull's whole allowance of simultaneous body transfers, handed out one permit per connection.
///
/// There are now TWO things that want to open connections — several FILES of a model at once
/// ([`crate::pull::fetch_all`]) and several RANGES of one file ([`crate::ranged`]) — and they must
/// not multiply. One budget shared by both is what makes the total bound `hub.pull_jobs` rather
/// than `pull_jobs × pull_jobs`: a 236-shard repo starts every permit on a file of its own, a
/// single-file repo spends them all on ranges of that file, and permits a finished file gives back
/// go to whichever files are still running.
///
/// Spare permits are handed out two ways. A file starting a ranged download takes a SHARE of what
/// is spare ([`Claim::take_share`]): files that start together divide it between them instead of
/// the first one taking it all. After that, a running download takes one more permit at each chunk
/// boundary ([`try_grow`](ConnBudget::try_grow)) — and not while a file is still waiting for its
/// share, which would otherwise be taken from under it.
///
/// Neither ever waits for a permit, so the file workers, which each hold one for their whole life,
/// can never be starved by range workers and nothing here can deadlock.
pub(crate) struct ConnBudget {
    free: AtomicUsize,
    /// What the budget was created with: `hub.pull_jobs`, at least one.
    capacity: usize,
    /// Files with a worker that have not yet taken their starting share or decided not to split.
    starting: AtomicUsize,
    /// Held while a share is sized and taken, so two files starting at once size theirs from
    /// counts the other has already updated.
    sharing: Mutex<()>,
}

/// One connection's worth of the budget, returned when dropped. Held by the thread that owns the
/// connection, so a panicking or early-returning worker cannot leak its slot.
pub(crate) struct Permit<'a> {
    budget: &'a ConnBudget,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.budget.free.fetch_add(1, Ordering::AcqRel);
    }
}

impl ConnBudget {
    pub(crate) fn new(permits: usize) -> Self {
        ConnBudget {
            free: AtomicUsize::new(permits),
            capacity: permits,
            starting: AtomicUsize::new(0),
            sharing: Mutex::new(()),
        }
    }

    /// The most connections this budget ever allows at once.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Permits available right now. Advisory only: another thread may take one a nanosecond later.
    pub(crate) fn available(&self) -> usize {
        self.free.load(Ordering::Acquire)
    }

    /// Register one file that is about to start, so that files starting together share what is
    /// spare (see [`Claim`]).
    pub(crate) fn claim(&self) -> Claim<'_> {
        self.starting.fetch_add(1, Ordering::AcqRel);
        Claim {
            budget: self,
            waiting: true,
        }
    }

    /// One more permit for a download that is already running, if one is free and no file is
    /// waiting for its starting share. Never blocks.
    ///
    /// A file that registered its claim after the check can still lose a permit to this call; it
    /// then starts with one less and grows at its own chunk boundaries like everyone else.
    pub(crate) fn try_grow(&self) -> Option<Permit<'_>> {
        if self.starting.load(Ordering::Acquire) > 0 {
            return None;
        }
        self.try_acquire()
    }

    /// Take one permit if one is free. Never blocks.
    pub(crate) fn try_acquire(&self) -> Option<Permit<'_>> {
        let mut free = self.free.load(Ordering::Acquire);
        loop {
            if free == 0 {
                return None;
            }
            match self.free.compare_exchange_weak(
                free,
                free - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(Permit { budget: self }),
                Err(now) => free = now,
            }
        }
    }

    /// Take up to `want` permits, returning however many were free. Never blocks, and returns an
    /// empty vector rather than waiting when the budget is exhausted.
    pub(crate) fn acquire_up_to(&self, want: usize) -> Vec<Permit<'_>> {
        let mut got = Vec::new();
        while got.len() < want {
            match self.try_acquire() {
                Some(p) => got.push(p),
                None => break,
            }
        }
        got
    }
}

/// One file's place among the files that are starting: registered by [`ConnBudget::claim`] when
/// a worker picks the file up, and given up when the file takes its share of the spare permits,
/// decides not to split, or is dropped.
///
/// The count is what makes the split fair. Files that start together are registered before any of
/// them takes a share, so each sizes its share against all of them rather than against whatever
/// happened to be free when it got there first.
pub(crate) struct Claim<'a> {
    budget: &'a ConnBudget,
    /// Still counted in `starting`.
    waiting: bool,
}

impl<'a> Claim<'a> {
    pub(crate) fn budget(&self) -> &'a ConnBudget {
        self.budget
    }

    /// This file's share of the spare permits, at most `want`: `ceil(spare / starting files)`,
    /// counting this one, and taken before this file stops counting as starting.
    ///
    /// Taken in turn, the shares add up to everything that was spare, so the last file of a group
    /// (or a file starting alone) leaves nothing idle. A file whose claim has already been given
    /// up — a download restarting after a changed object — counts itself in again for the sizing.
    pub(crate) fn take_share(&mut self, want: usize) -> Vec<Permit<'a>> {
        let budget = self.budget;
        let _turn = budget.sharing.lock().unwrap_or_else(|e| e.into_inner());
        let sharers = budget.starting.load(Ordering::Acquire) + usize::from(!self.waiting);
        let share = budget.available().div_ceil(sharers).min(want);
        let got = budget.acquire_up_to(share);
        self.stand_aside();
        got
    }

    /// Stop counting as a starting file without taking anything: the file is not being split, or
    /// is about to wait for something that may take arbitrarily long.
    pub(crate) fn stand_aside(&mut self) {
        if std::mem::take(&mut self.waiting) {
            self.budget.starting.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.stand_aside();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The budget hands out exactly what it has and not one more, and every permit comes back when
    /// it drops. A budget that over-issued would show up in `infr pull` as more sockets than
    /// `hub.pull_jobs` — the bound the whole fan-out rests on.
    #[test]
    fn a_budget_issues_exactly_its_permits() {
        let b = ConnBudget::new(3);
        let mut held = b.acquire_up_to(10);
        assert_eq!(held.len(), 3, "asked for 10 of 3");
        assert_eq!(b.available(), 0);
        assert!(b.try_acquire().is_none(), "over-issued");
        held.pop();
        assert_eq!(b.available(), 1);
        assert!(b.try_acquire().is_some());
        drop(held);
        assert_eq!(b.available(), 3, "permits must return when dropped");
    }

    /// `HF_TOKEN=` is no token at all, not an empty bearer credential, and surrounding whitespace
    /// is not sent.
    #[test]
    fn an_empty_token_is_no_token() {
        assert_eq!(usable_token(None), None);
        assert_eq!(usable_token(Some(String::new())), None);
        assert_eq!(usable_token(Some("  ".into())), None);
        // The value sent is the trimmed one: a trailing newline from an env file is not part of
        // the token.
        assert_eq!(
            usable_token(Some(" hf_x\n".into())).as_deref(),
            Some("hf_x")
        );
    }

    /// `hub.pull_jobs = 0`/`1` leave nothing over for range workers, which is what keeps those
    /// settings STRICTLY one connection (see `HubCfg::pull_jobs`).
    #[test]
    fn an_empty_budget_grants_nothing() {
        let b = ConnBudget::new(0);
        assert_eq!(b.available(), 0);
        assert!(b.try_acquire().is_none());
        assert!(b.acquire_up_to(4).is_empty());
        assert!(b.claim().take_share(4).is_empty());
    }

    /// Three files starting on eight connections hold three permits; the five spare are split
    /// 2 + 2 + 1 in whatever order the files get there, and nothing is left idle. A first-come
    /// split would hand the first file all five.
    #[test]
    fn files_starting_together_split_the_spare() {
        let b = ConnBudget::new(8);
        let _workers = b.acquire_up_to(3);
        let mut claims: Vec<_> = (0..3).map(|_| b.claim()).collect();
        let shares: Vec<_> = claims.iter_mut().map(|c| c.take_share(100)).collect();
        let sizes: Vec<_> = shares.iter().map(Vec::len).collect();
        assert_eq!(sizes, [2, 2, 1]);
        assert_eq!(b.available(), 0);
        // A share never exceeds what the file can use.
        drop(shares);
        assert_eq!(b.claim().take_share(2).len(), 2);
    }

    /// A running download cannot grow while a file is still waiting for its starting share — it
    /// would take that share from under it — and can as soon as the file has taken it or stood
    /// aside.
    #[test]
    fn growth_waits_for_starting_files() {
        let b = ConnBudget::new(4);
        let mut claim = b.claim();
        assert!(b.try_grow().is_none(), "grew past a waiting file");
        claim.stand_aside();
        assert!(b.try_grow().is_some());
        let late = b.claim();
        assert!(b.try_grow().is_none());
        drop(late);
        assert!(b.try_grow().is_some(), "a dropped claim must stop counting");
    }

    /// A claim that already stood aside (a restarted download) counts itself in again, so alone it
    /// takes everything spare rather than dividing by zero.
    #[test]
    fn a_spent_claim_takes_everything_alone() {
        let b = ConnBudget::new(4);
        let mut claim = b.claim();
        claim.stand_aside();
        assert_eq!(claim.take_share(10).len(), 4);
    }
}
