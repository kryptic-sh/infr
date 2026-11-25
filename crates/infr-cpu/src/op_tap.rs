//! A test tap on the shape of the graphs the CPU executes: the row count of every
//! `Op::Attention` and `Op::Mla`, the shape of every `Op::MulVec`, and the row count and table
//! dtype of every `Op::EmbedGather`. Off unless a test starts it on the thread that executes the
//! graphs, and then it only copies what the ops carry anyway. It exists so a test can see how a
//! prompt was cut into graphs (one row per graph, or chunks of several), whether the embedding was
//! gathered in the graph, and which ops a model emitted around its attention, which the outputs
//! cannot show when the two ways are meant to compute the same bits.

use infr_core::DType;
use std::cell::RefCell;

/// One op the tap saw, in execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TappedOp {
    /// An `Op::Attention` over `rows` query rows.
    Attention { rows: u32 },
    /// An `Op::EmbedGather` of `rows` rows out of a table stored as `table`.
    EmbedGather { rows: u32, table: DType },
    /// An `Op::Mla` over `rows` query rows.
    Mla { rows: u32 },
    /// An `Op::MulVec` scaling `rows` rows of `n`.
    MulVec { rows: u32, n: u32 },
}

thread_local! {
    static OPS: RefCell<Option<Vec<TappedOp>>> = const { RefCell::new(None) };
}

/// Start recording on this thread, dropping anything recorded before.
pub fn start() {
    OPS.with(|o| *o.borrow_mut() = Some(Vec::new()));
}

/// Stop recording and return every op recorded since [`start`], in execution order.
pub fn take() -> Vec<TappedOp> {
    OPS.with(|o| o.borrow_mut().take().unwrap_or_default())
}

pub(crate) fn record(op: TappedOp) {
    OPS.with(|o| {
        if let Some(v) = o.borrow_mut().as_mut() {
            v.push(op);
        }
    });
}
