//! A test tap on the CPU `Op::LightningIndexer`: what each query row scored and which keys it
//! kept. Off unless a test starts it on the thread that executes the graph, and then it only
//! copies what the op computes anyway. It exists so a test can compare the indexer's selection
//! with another engine's row by row, which a model's logits cannot show: a selection at a row
//! no later row reads leaves every logit unmoved.

use std::cell::RefCell;

/// One query row an `Op::LightningIndexer` scored.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexerRow {
    /// Which op scored the row: the ops recorded since [`start`], counted from 0 in execution
    /// order. A graph runs one op per indexed layer, each over the graph's rows, so this is what
    /// tells two layers' rows of one position apart once a graph holds several positions.
    pub op: usize,
    /// The row's absolute position.
    pub pos: usize,
    /// The `top_k` keys it kept, in the op's output order (eligible keys by score, then any
    /// ineligible fill).
    pub kept: Vec<u32>,
    /// The score of every eligible key, by key index.
    pub scores: Vec<f32>,
}

thread_local! {
    static ROWS: RefCell<Option<Vec<IndexerRow>>> = const { RefCell::new(None) };
    static OPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Start recording on this thread, dropping anything recorded before.
pub fn start() {
    ROWS.with(|r| *r.borrow_mut() = Some(Vec::new()));
    OPS.with(|o| o.set(0));
}

/// Stop recording and return every row recorded since [`start`], in execution order: op by op,
/// and within an op by row.
pub fn take() -> Vec<IndexerRow> {
    ROWS.with(|r| r.borrow_mut().take().unwrap_or_default())
}

pub(crate) fn active() -> bool {
    ROWS.with(|r| r.borrow().is_some())
}

/// Record one op's rows, stamping each with the op's ordinal ([`IndexerRow::op`]); `rows` is
/// handed that ordinal to build them with.
pub(crate) fn record<I: Iterator<Item = IndexerRow>>(rows: impl FnOnce(usize) -> I) {
    ROWS.with(|r| {
        if let Some(v) = r.borrow_mut().as_mut() {
            let op = OPS.with(|o| o.replace(o.get() + 1));
            v.extend(rows(op));
        }
    });
}
