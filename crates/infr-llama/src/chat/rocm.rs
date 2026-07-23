//! ROCm-backed [`ChatModel`] (Linux + ROCm/HIP only): the AMD-GPU twin of
//! [`crate::chat::MetalSeamChat`]: weights upload once, the KV cache persists across
//! turns, and each turn prefills only the suffix that differs from the previous rendered
//! history.
//!
//! The real implementation lives behind `cfg(all(target_os = "linux", feature = "rocm"))`;
//! without the feature the constructor returns a clean error message.

use super::ChatModel;
use crate::{GenStats, SeamModel};

#[cfg(all(target_os = "linux", feature = "rocm"))]
use crate::seam::model::DenseRocmSession;
#[cfg(all(target_os = "linux", feature = "rocm"))]
use anyhow::Result;

/// ROCm seam backend — the AMD-GPU twin of [`MetalSeamChat`]: persistent session,
/// KV cache across turns, suffix-only prefill.
#[cfg(all(target_os = "linux", feature = "rocm"))]
pub struct RocmSeamChat {
    model: SeamModel,
    session: Option<DenseRocmSession>,
    dev_idx: u32,
}

#[cfg(all(target_os = "linux", feature = "rocm"))]
impl RocmSeamChat {
    pub fn new(model: SeamModel, dev_idx: u32) -> Result<Self> {
        Ok(Self {
            model,
            session: None,
            dev_idx,
        })
    }

    fn ensure_session(&mut self) -> Result<()> {
        if self.session.is_none() {
            let train = self.model.config().n_ctx_train;
            let max_ctx = std::env::var("INFR_CTX")
                .ok()
                .and_then(|v| infr_core::parse_size(&v))
                .map(|s| s.resolve(train as u64) as usize)
                .unwrap_or(train);
            self.session = Some(self.model.rocm_session(max_ctx, self.dev_idx)?);
        }
        Ok(())
    }
}

#[cfg(all(target_os = "linux", feature = "rocm"))]
impl ChatModel for RocmSeamChat {
    fn render_model(&self) -> &SeamModel {
        &self.model
    }

    fn reset_kv(&mut self) {
        if let Some(s) = &mut self.session {
            s.reset_cache();
        }
    }

    fn warmup(&mut self) -> Result<()> {
        self.generate("Hi", 2, None, &mut |_| {})?;
        if let Some(s) = &mut self.session {
            s.reset_cache();
        }
        Ok(())
    }

    fn generate(
        &mut self,
        _prompt: &str,
        _max_new: usize,
        _req: Option<&crate::sampling::RequestCtx>,
        _on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.ensure_session()?;
        // The ROCm dense seam runner (`generate_dense_rocm`) is not yet implemented —
        // this path is wired for session management only. Phase 2 will add the runner.
        anyhow::bail!("ROCm dense generation not yet implemented — session is active (Phase 2)")
    }
}

// ── Placeholder (feature not active) ─────────────────────────────────────────

/// ROCm seam backend placeholder — returns a clean error from `new()` so the CLI
/// can surface it as a build-time feature gate.
#[cfg(not(all(target_os = "linux", feature = "rocm")))]
pub struct RocmSeamChat {
    #[allow(dead_code)]
    _model: SeamModel,
}

#[cfg(not(all(target_os = "linux", feature = "rocm")))]
impl RocmSeamChat {
    pub fn new(_model: SeamModel, _dev_idx: u32) -> anyhow::Result<Self> {
        anyhow::bail!(
            "ROCm backend not compiled — build with `cargo build --features rocm` \
             on a Linux machine with ROCm/HIP installed (docs/rocm-plan.md Phase 0)"
        )
    }
}

#[cfg(not(all(target_os = "linux", feature = "rocm")))]
impl ChatModel for RocmSeamChat {
    fn render_model(&self) -> &SeamModel {
        unreachable!("RocmSeamChat placeholder")
    }

    fn reset_kv(&mut self) {}

    fn warmup(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    fn generate(
        &mut self,
        _prompt: &str,
        _max_new: usize,
        _req: Option<&crate::sampling::RequestCtx>,
        _on_piece: &mut dyn FnMut(&str),
    ) -> anyhow::Result<GenStats> {
        unreachable!("RocmSeamChat::generate: backend not compiled")
    }
}
