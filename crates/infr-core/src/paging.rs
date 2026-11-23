//! Typed refusals shared by the model layer and the server when a paged KV budget runs out.
//!
//! The pool is the one place a request can be turned away before it touches the GPU, and the
//! reason belongs to the request (it asked for more pages than the session has), not to the
//! server's own health — so it travels as a downcastable error and the server answers `400`.

use std::fmt;

/// A slot's page map could not grow: the session's physical page budget is spent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPagesExhausted {
    /// Logical pages the request needs in total.
    pub need: usize,
    /// Logical pages this slot had already allocated.
    pub allocated: usize,
    /// Physical pages free when the allocation was refused.
    pub free: usize,
    /// Physical pages the session's pool holds, the zero page included.
    pub total: usize,
}

impl fmt::Display for KvPagesExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "kv.paged: the pool has {} free of {} pages, and this request needs {} ({} already \
             allocated)",
            self.free, self.total, self.need, self.allocated
        )
    }
}

impl std::error::Error for KvPagesExhausted {}
