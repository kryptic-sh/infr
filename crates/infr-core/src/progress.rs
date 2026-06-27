//! Shared progress-bar style (thin wrapper over `indicatif`) so every long operation — model
//! downloads, weight loading, … — looks identical. Use [`bar`] wherever you need progress.
//!
//! Layout is a single auto-width line: details on the left/right, the bar fills the middle:
//! ```text
//! label  37/40 layers [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 3s
//! label  4.2GB/8.4GB  [━━━━━━━━━━━━━━━━╾────────────] 180MB/s ETA 23s
//! ```

use indicatif::{ProgressBar, ProgressStyle};
use std::io::IsTerminal;

/// What the bar counts — controls the left/right stat fields (the layout is identical either way).
pub enum Unit {
    /// Byte transfer: `done/total … rate ETA` (e.g. downloads).
    Bytes,
    /// A count of discrete items named by the string (e.g. `Items("layers")`).
    Items(&'static str),
}

/// A consistently-styled, auto-width progress bar.
///
/// - `total: Some(n)` → a full-width bar; `None` → a spinner (unknown length).
/// - `label` is the left-most field (the `{msg}`).
/// - Hidden when stderr isn't a TTY (piped output, `infr serve`) so it never spams logs.
pub fn bar(total: Option<u64>, label: &str, unit: Unit) -> ProgressBar {
    if !std::io::stderr().is_terminal() {
        return ProgressBar::hidden();
    }
    // Single line: `{msg} <left> [{wide_bar}] <right>` — wide_bar absorbs all free terminal width.
    let (left, right) = match unit {
        Unit::Bytes => (
            "{bytes}/{total_bytes}".to_string(),
            "{bytes_per_sec} ETA {eta}",
        ),
        Unit::Items(name) => (format!("{{pos}}/{{len}} {name}"), "{elapsed}"),
    };
    let pb = match total {
        Some(n) => {
            let pb = ProgressBar::new(n);
            pb.set_style(
                ProgressStyle::with_template(&format!(
                    "{{msg}}  {left} [{{wide_bar:.cyan/blue}}] {right}"
                ))
                .unwrap()
                .progress_chars("━━╾─"),
            );
            pb
        }
        None => {
            // Unknown length → spinner (no bar to fill); same left/right fields.
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template(&format!(
                    "{{msg}}  {{spinner:.green}} {left} {right}"
                ))
                .unwrap(),
            );
            pb
        }
    };
    pb.set_message(label.to_owned());
    pb
}
