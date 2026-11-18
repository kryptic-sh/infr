//! How far a backend's logits are from the CPU reference's, and how far they may be.
//!
//! The CPU backend's output is bit-identical across systems, tiers, thread counts and chunk
//! heights (`docs/plan/11-cpu-bit-identical.md`), so it is the one reference every GPU is held
//! to. GPU kernels are not bit-identical to it, and become tunable per device, so the check is a
//! tolerance: the backend under test is teacher-forced along the reference's own greedy tokens
//! (the two never fork at a near tie), and each scored position's two logit rows are compared by
//! [`metrics::score_row`]. See `docs/gpu-tolerance.md`.

pub mod metrics;
pub mod table;

pub use metrics::{score_row, RowError, RowScore, RunScore};
pub use table::{tolerance, Entry, PathClass, Tolerance, FIRST_ROWS, MIN_TOP1_SCORED, TABLE};
