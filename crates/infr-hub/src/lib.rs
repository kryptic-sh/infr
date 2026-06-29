//! Model acquisition: resolve an `hf:org/repo[:quant]` reference (or a plain path) to a local GGUF,
//! pulling from HuggingFace over plain HTTP (no external CLI) with resume + a progress bar.
//!
//! Models live in the **standard HF Hub cache** (`~/.cache/huggingface/hub`), shared with llama.cpp
//! and `huggingface_hub`, so `infr run hf:org/repo:Q4_K_M` and `llama-cli -hf org/repo:Q4_K_M` use
//! the same files — see [`store`] for the layout.

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
