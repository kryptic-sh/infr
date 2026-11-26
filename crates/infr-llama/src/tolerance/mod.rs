//! How far a backend's logits are from the CPU reference's, and how far they may be.
//!
//! The CPU backend's output is bit-identical across systems, tiers, thread counts and chunk
//! heights (`docs/plan/11-cpu-bit-identical.md`), so it is the one reference every GPU is held
//! to. GPU kernels are not bit-identical to it, and become tunable per device, so the check is a
//! tolerance: the backend under test is teacher-forced along the reference's own greedy tokens
//! (the two never fork at a near tie), and each scored position's two logit rows are compared by
//! [`metrics::score_row`]. A routed-expert (MoE) model is forced along the reference's expert
//! selections as well, so the rows compare the two backends' arithmetic under one routing, and
//! what each router would have selected itself is compared on its own
//! ([`metrics::RouteScore`]). See `docs/gpu-tolerance.md`.
//!
//! [`metrics`] has the distances, [`table`] the bounds, and [`gate`] and [`greedy`] the
//! evaluation that holds a model to them: what the GPU tolerance tests and `infr tune check` run.
//! [`oracle`] keeps the CPU reference those tests score against in an on-disk cache.

pub mod gate;
pub mod greedy;
pub mod metrics;
pub mod oracle;
pub mod table;

pub use metrics::{score_row, RouteFlip, RouteScore, RowError, RowScore, RunScore, TokenScore};
pub use table::{
    lookup, tolerance, Check, Entry, Limit, PathClass, Tolerance, FIRST_ROWS, MIN_TOP1_SCORED,
    TABLE,
};
