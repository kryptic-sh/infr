//! `infr-plat` — the platform seam.
//!
//! One capability per module, each with an arm per operating system, so the OS-specific surface
//! of the workspace is a single crate a reviewer can read end to end. Everything here answers the
//! same shape of question: *how do I spell this on the host I was compiled for?* — a file lock, a
//! positioned read, the amount of memory left, whether a pid is alive.
//!
//! **A leaf, deliberately.** `infr-plat` depends on no other `infr-*` crate and must not grow
//! one: the signal handler takes its shutdown latch as a `fn` pointer rather than calling into
//! `infr-core` precisely so this stays true. Nothing here knows about tensors, backends or
//! models.
//!
//! **Not a hardware abstraction layer.** GPU backends plug in through
//! `infr_core::backend::Backend`; `infr-metal` is a backend and stays where it is. This crate is
//! about the OS, not the device.
//!
//! # What is NOT unified
//!
//! Giving two different guarantees one signature is how "unsupported here" arrives as "passed",
//! so where the platforms genuinely differ, the difference is in the API or in the doc comment
//! rather than sanded off:
//!
//! - [`mem::available_bytes`] is clamped by the cgroup limit on Linux, unclamped on Windows, and
//!   absent on macOS — [`mem::Available`] reports which, because sizing an arena from an
//!   unclamped figure inside a container is an OOM kill.
//! - [`lock::FileLock`] is advisory on Unix and mandatory on Windows. It promises the weaker of
//!   the two.
//! - [`proc::pid_alive`] answers `true` on a platform with no probe — losing a tripwire rather
//!   than misreporting a live process as dead.

pub mod fileio;
pub mod link;
pub mod lock;
pub mod paths;
pub mod proc;
pub mod signal;
