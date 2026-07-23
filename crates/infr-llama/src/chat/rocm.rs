//! ROCm-backed [`ChatModel`] (Linux + ROCm/HIP only): placeholder for Phase 0 —
//! the backend is selectable via `--dev rocm` but reports a clean "not yet implemented"
//! error since the kernels haven't been written yet. Real implementation lives behind
//! `cfg(feature = "rocm")` once the HIP FFI is wired.

use super::ChatModel;
use crate::{GenStats, SeamModel};
use anyhow::Result;

/// ROCm seam backend placeholder — returns a clean "not yet implemented" error from
/// `new()` so the CLI can surface it as a build-time feature gate.
pub struct RocmSeamChat {
    _model: SeamModel,
}

impl RocmSeamChat {
    pub fn new(_model: SeamModel) -> Result<Self> {
        // Phase 0: the crate scaffolding exists but the HIP FFI is not wired yet.
        // `cargo build --features rocm` on a Linux box with ROCm installed will
        // compile the real backend; without the feature, this constructor errors.
        anyhow::bail!(
            "ROCm backend not compiled — build with `cargo build --features rocm` \
             on a Linux machine with ROCm/HIP installed (docs/rocm-plan.md Phase 0)"
        )
    }
}

impl ChatModel for RocmSeamChat {
    fn render_model(&self) -> &SeamModel {
        &self._model
    }

    fn reset_kv(&mut self) {}

    fn warmup(&mut self) -> Result<()> {
        Ok(())
    }

    fn generate(
        &mut self,
        _prompt: &str,
        _max_new: usize,
        _req: Option<&crate::sampling::RequestCtx>,
        _on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        unreachable!("RocmSeamChat::generate: backend not compiled")
    }
}
