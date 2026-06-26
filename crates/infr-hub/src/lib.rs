//! Model acquisition: resolve a reference to a local GGUF, reusing/extending the **Ollama
//! store** (same dir + format) so existing Ollama downloads work with zero re-download.
//!
//! See PLAN.md §"fetch / model acquisition (infr-hub)". Store layout:
//!
//! ```text
//! $INFR_MODELS | $OLLAMA_MODELS | ~/.ollama/models
//!   manifests/<registry>/<ns>/<name>/<tag>   (OCI-style JSON)
//!   blobs/sha256-<digest>                    (layer blobs; model layer == GGUF)
//! ```

mod model_ref;
mod pull;
mod store;

pub use model_ref::ModelRef;
pub use pull::pull;
pub use store::Store;

use infr_core::error::Result;
use std::path::PathBuf;

/// Resolve from the store if present, otherwise pull.  Used by `infr run` / `infr serve`.
///
/// - `Path(p)` → returned immediately.
/// - Everything else → [`Store::discover`] + [`Store::resolve`]; if not cached → [`pull`].
pub fn ensure(r: &ModelRef) -> Result<PathBuf> {
    if let ModelRef::Path(p) = r {
        return Ok(p.clone());
    }
    let store = Store::discover()?;
    if let Some(p) = store.resolve(r)? {
        return Ok(p);
    }
    pull(r)
}
