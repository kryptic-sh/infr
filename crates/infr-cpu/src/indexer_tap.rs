//! A test tap on the CPU `Op::LightningIndexer`: what each query row scored and which keys it
//! kept. Off unless a test starts it on the thread that executes the graph, and then it only
//! copies what the op computes anyway. It exists so a test can compare the indexer's selection
//! with another engine's row by row, which a model's logits cannot show: a selection at a row
//! no later row reads leaves every logit unmoved.

use std::cell::RefCell;

/// One query row an `Op::LightningIndexer` scored.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexerRow {
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
}

/// Start recording on this thread, dropping anything recorded before.
pub fn start() {
    ROWS.with(|r| *r.borrow_mut() = Some(Vec::new()));
}

/// Stop recording and return every row recorded since [`start`], in execution order: op by op,
/// and within an op by row.
pub fn take() -> Vec<IndexerRow> {
    ROWS.with(|r| r.borrow_mut().take().unwrap_or_default())
}

pub(crate) fn active() -> bool {
    ROWS.with(|r| r.borrow().is_some())
}

pub(crate) fn record(rows: impl Iterator<Item = IndexerRow>) {
    ROWS.with(|r| {
        if let Some(v) = r.borrow_mut().as_mut() {
            v.extend(rows);
        }
    });
}
