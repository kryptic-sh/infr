//! ROCm-backed [`ChatModel`] (Linux + ROCm/HIP only): the AMD-GPU twin of
//! [`crate::chat::MetalSeamChat`]: weights upload once, the KV cache persists across
//! turns, and each turn prefills only the suffix that differs from the previous rendered
//! history.
//!
//! The real implementation lives behind `cfg(all(target_os = "linux", feature = "rocm"))`;
//! without the feature the constructor returns a clean error message.

use super::ChatModel;
#[cfg(all(target_os = "linux", feature = "rocm"))]
use crate::seam::model::DenseRocmSession;
#[cfg(all(target_os = "linux", feature = "rocm"))]
use crate::{GenStats, SeamModel};
#[cfg(all(target_os = "linux", feature = "rocm"))]
use anyhow::Result;

/// ROCm seam backend — the AMD-GPU twin of [`MetalSeamChat`]: persistent session,
/// KV cache across turns, suffix-only prefill.
#[cfg(all(target_os = "linux", feature = "rocm"))]
pub struct RocmSeamChat {
    model: SeamModel,
    session: Option<DenseRocmSession>,
}

#[cfg(all(target_os = "linux", feature = "rocm"))]
impl RocmSeamChat {
    pub fn new(model: SeamModel) -> Result<Self> {
        Ok(Self { model, session: None })
    }

    fn ensure_session(&mut self) -> Result<()> {
        if self.session.is_none() {
            let train = self.model.config().n_ctx_train;
            let max_ctx = std::env::var("INFR_CTX")
                .ok()
                .and_then(|v| infr_core::parse_size(&v))
                .map(|s| s.resolve(train as u64) as usize)
                .unwrap_or(train);
            self.session = Some(self.model.rocm_session(max_ctx)?);
        }
        Ok(())
    }
}

#[cfg(all(target_os = "linux", feature = "rocm"))]
impl ChatModel for RocmSeamChat {
    fn render_model(&self) -> &SeamModel { &self.model }
    fn reset_kv(&mut self) { if let Some(s) = &mut self.session { s.reset_cache(); } }
    fn warmup(&mut self) -> Result<()> {
        self.generate("Hi", 2, None, &mut |_| {})?;
        if let Some(s) = &mut self.session { s.reset_cache(); }
        Ok(())
    }
    fn generate(&mut self, _prompt: &str, _max_new: usize, _req: Option<&crate::sampling::RequestCtx>, _on_piece: &mut dyn FnMut(&str)) -> Result<GenStats> {
        self.ensure_session()?;
        anyhow::bail!("ROCm dense generation not yet implemented — session is active (Phase 2)")
    }
}

// ── Placeholder (feature not active) ─────────────────────────────────────────

#[cfg(not(all(target_os = "linux", feature = "rocm")))]
pub struct RocmSeamChat {
    #[allow(dead_code)]
    model: crate::SeamModel,
}

#[cfg(not(all(target_os = "linux", feature = "rocm")))]
impl RocmSeamChat {
    pub fn new(_model: crate::SeamModel) -> anyhow::Result<Self> {
        anyhow::bail!("ROCm backend not compiled — build with `cargo build --features rocm` on a Linux machine with ROCm/HIP installed (docs/rocm-plan.md Phase 0)")
    }
}

#[cfg(not(all(target_os = "linux", feature = "rocm")))]
impl ChatModel for RocmSeamChat {
    fn render_model(&self) -> &crate::SeamModel { &self.model }
    fn reset_kv(&mut self) {}
    fn warmup(&mut self) -> anyhow::Result<()> { Ok(()) }
    fn generate(&mut self, _prompt: &str, _max_new: usize, _req: Option<&crate::sampling::RequestCtx>, _on_piece: &mut dyn FnMut(&str)) -> anyhow::Result<crate::GenStats> {
        unreachable!("RocmSeamChat::generate: backend not compiled")
    }
}
