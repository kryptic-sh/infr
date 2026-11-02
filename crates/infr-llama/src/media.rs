//! Multimodal prompt plumbing: images ride NEXT TO the rendered prompt string, not inside it.
//!
//! The chat surface (`ChatModel::generate`, `ParallelSeam::generate`, …) speaks rendered prompt
//! strings, and threading an image list through every one of those signatures — and the 22 callers
//! of the seam runner — would touch far more than this feature needs. Instead:
//!
//! 1. The server renders each image part as [`IMAGE_MARKER`] in the message text, in order, and the
//!    caller enters a [`MediaScope`] holding the image bytes for the duration of the generation.
//! 2. [`crate::SeamModel::encode_prompt`] splits the rendered prompt on the marker, runs the vision
//!    encoder, and lays each image out as `<|image>` + N placeholder ids + `<image|>`. The
//!    placeholder ids sit above every vocab ([`MEDIA_ID_BASE`]) and are derived from the image's
//!    content hash, so the KV prefix cache reuses an image's rows only when the image is the same.
//! 3. It returns an [`OverrideScope`] that publishes the encoded rows for the runner, which swaps
//!    them in for the placeholders' embeddings during prefill (`seam::runner`).
//!
//! Both scopes are thread-local: every serve request runs its whole generation on one thread (the
//! multi-slot engine takes turns on a baton, it does not hop threads), and the scope guards restore
//! the previous value on drop.

use std::cell::RefCell;
use std::sync::Arc;

pub use infr_chat::IMAGE_MARKER;

/// Token ids at or above this are image placeholders, never real vocabulary.
pub const MEDIA_ID_BASE: u32 = 0x8000_0000;

/// True for an image placeholder id (see [`MEDIA_ID_BASE`]).
#[inline]
pub fn is_media_id(id: u32) -> bool {
    id >= MEDIA_ID_BASE
}

/// The current thread's images (see [`MediaScope`]).
type Images = Arc<Vec<Arc<[u8]>>>;

thread_local! {
    static MEDIA: RefCell<Option<Images>> = const { RefCell::new(None) };
    static OVERRIDE: RefCell<Option<Arc<EmbedOverride>>> = const { RefCell::new(None) };
}

/// RAII: the images (raw PNG/JPEG/GIF/WebP bytes) for the prompt generated on this thread, in the
/// order their [`IMAGE_MARKER`]s appear.
pub struct MediaScope {
    prev: Option<Images>,
}

impl MediaScope {
    pub fn enter(images: Vec<Arc<[u8]>>) -> Self {
        let prev = MEDIA.with(|m| m.borrow_mut().replace(Arc::new(images)));
        Self { prev }
    }
}

impl Drop for MediaScope {
    fn drop(&mut self) {
        MEDIA.with(|m| *m.borrow_mut() = self.prev.take());
    }
}

pub(crate) fn current_media() -> Option<Images> {
    MEDIA.with(|m| m.borrow().clone())
}

/// Embedding rows that replace the token embedding at given prompt positions.
pub(crate) struct EmbedOverride {
    pub(crate) n_embd: usize,
    /// `(first position, rows [len * n_embd])`, sorted by position, non-overlapping.
    pub(crate) spans: Vec<(usize, Arc<Vec<f32>>)>,
}

impl EmbedOverride {
    fn span_len(&self, rows: &[f32]) -> usize {
        rows.len() / self.n_embd
    }

    /// The override row for `pos`, if any.
    pub(crate) fn row(&self, pos: usize) -> Option<&[f32]> {
        self.spans.iter().find_map(|(start, rows)| {
            let len = self.span_len(rows);
            (pos >= *start && pos < start + len)
                .then(|| &rows[(pos - start) * self.n_embd..(pos - start + 1) * self.n_embd])
        })
    }

    /// Whether any overridden position falls in `range`.
    pub(crate) fn any_in(&self, range: std::ops::Range<usize>) -> bool {
        self.spans.iter().any(|(start, rows)| {
            let end = start + self.span_len(rows);
            *start < range.end && range.start < end
        })
    }
}

/// RAII: publishes an [`EmbedOverride`] for the runner on this thread.
pub(crate) struct OverrideScope {
    prev: Option<Arc<EmbedOverride>>,
}

impl OverrideScope {
    pub(crate) fn enter(o: EmbedOverride) -> Self {
        let prev = OVERRIDE.with(|c| c.borrow_mut().replace(Arc::new(o)));
        Self { prev }
    }
}

impl Drop for OverrideScope {
    fn drop(&mut self) {
        OVERRIDE.with(|c| *c.borrow_mut() = self.prev.take());
    }
}

pub(crate) fn current_override() -> Option<Arc<EmbedOverride>> {
    OVERRIDE.with(|c| c.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_lookup_and_ranges() {
        let o = EmbedOverride {
            n_embd: 2,
            spans: vec![
                (3, Arc::new(vec![1., 2., 3., 4.])),
                (10, Arc::new(vec![5., 6.])),
            ],
        };
        assert_eq!(o.row(2), None);
        assert_eq!(o.row(3), Some(&[1., 2.][..]));
        assert_eq!(o.row(4), Some(&[3., 4.][..]));
        assert_eq!(o.row(5), None);
        assert_eq!(o.row(10), Some(&[5., 6.][..]));
        assert!(!o.any_in(0..3));
        assert!(o.any_in(0..4));
        assert!(o.any_in(4..6));
        assert!(!o.any_in(5..10));
        assert!(o.any_in(10..11));
    }

    #[test]
    fn scopes_nest_and_restore() {
        assert!(current_media().is_none());
        {
            let _a = MediaScope::enter(vec![Arc::from(&b"a"[..])]);
            {
                let _b = MediaScope::enter(vec![]);
                assert_eq!(current_media().unwrap().len(), 0);
            }
            assert_eq!(current_media().unwrap().len(), 1);
        }
        assert!(current_media().is_none());
    }
}
