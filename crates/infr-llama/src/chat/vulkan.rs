//! Vulkan-backed [`ChatModel`]: [`DenseSeamChat`] (dense/MoE — and, since the phase-3 cutover,
//! qwen35 too — on the Vulkan agnostic seam with a persistent KV session).

use super::ChatModel;
use crate::{GenStats, SeamModel};
use anyhow::Result;

/// Dense/MoE on the VULKAN agnostic seam with a persistent KV session (`INFR_SEAM=1` for
/// `infr run`): weights upload once, and every turn prefills only the token suffix that differs
/// from the previous turn — the seam twin of the bespoke `ChatSession`'s incremental prefill.
///
/// This is the default `infr run`/`infr serve` path for EVERY arch including qwen35 (Phase 3
/// cutover — see the matching comment at both CLI call sites), so it's also where MTP mode
/// (issue #33, `docs/MTP.md`) lives: `mtp_head` is `Some` once resolved+loaded, built lazily on
/// the first [`generate`](ChatModel::generate) call when [`wants_mtp`](Self::wants_mtp) is true
/// (opt-in `INFR_MTP=1`, and only for a qwen35 GGUF that actually ships an MTP head —
/// `Config::n_layer_nextn`'s doc). `INFR_MTP` unset/`0`, or a GGUF without an MTP head:
/// `wants_mtp` is always false, `mtp_head` stays `None` forever, and `generate` takes the EXACT
/// same `session` path it always has — zero risk to non-MTP models/GGUFs.
pub struct DenseSeamChat {
    model: SeamModel,
    session: Option<crate::seam::model::DenseVulkanSession>,
    mtp_head: Option<crate::mtp::MtpHeadWeights>,
    mtp_checked: bool,
}

impl DenseSeamChat {
    pub fn new(model: SeamModel) -> Self {
        Self {
            model,
            session: None,
            mtp_head: None,
            mtp_checked: false,
        }
    }

    /// MTP mode is opt-in (`INFR_MTP=1`) and Vulkan-only this phase (the invariant test + the
    /// oracle comparison in `docs/MTP.md` are both pinned on Vulkan — CPU/Metal MTP is
    /// unimplemented, not merely untested; `DenseSeamChat` IS always Vulkan, so no backend gate
    /// is needed here beyond the GGUF check). Memoized after the first call (`mtp_checked`) so a
    /// non-MTP GGUF doesn't re-parse its `Config` every turn.
    fn wants_mtp(&mut self) -> Result<bool> {
        if self.mtp_head.is_some() {
            return Ok(true);
        }
        if self.mtp_checked {
            return Ok(false);
        }
        self.mtp_checked = true;
        if std::env::var("INFR_MTP").ok().as_deref() != Some("1") {
            return Ok(false);
        }
        if self.model.config().n_layer_nextn == 0 {
            return Ok(false);
        }
        self.mtp_head = Some(crate::mtp::load_mtp_head(
            self.model.gguf(),
            self.model.config(),
        )?);
        Ok(true)
    }

    /// Lazily open the persistent Vulkan session. Explicit `INFR_MAX_CTX` = user override, used
    /// verbatim (NEVER clamped — the Vulkan VRAM budget guard still errors cleanly at alloc time
    /// if it truly doesn't fit); unset = the model's trained context, clamped to the VRAM budget
    /// (`vulkan_session_default`) so a long-context model's default KV cache can't blow VRAM.
    fn ensure_session(&mut self) -> Result<()> {
        if self.session.is_none() {
            let user_ctx: Option<usize> = std::env::var("INFR_MAX_CTX")
                .ok()
                .and_then(|v| v.parse().ok());
            self.session = Some(match user_ctx {
                Some(ctx) => self.model.vulkan_session(ctx)?,
                None => self.model.vulkan_session_default()?,
            });
        }
        Ok(())
    }
}

impl ChatModel for DenseSeamChat {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.model.render_chat_messages(messages)
    }

    fn warmup(&mut self) -> Result<()> {
        let prof2 = std::env::var_os("INFR_PROF2");
        if prof2.is_some() {
            std::env::remove_var("INFR_PROF2");
        }
        let r = self.generate("Hi", 2, &mut |_| {});
        if let Some(v) = prof2 {
            std::env::set_var("INFR_PROF2", v);
        }
        r?;
        // Drop the warmup tokens so the first real prompt prefills clean slots from row 0
        // instead of forking off a garbage prefix.
        if let Some(s) = &mut self.session {
            s.reset_cache();
        }
        Ok(())
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.wants_mtp()? {
            let head = self.mtp_head.as_ref().expect("wants_mtp loaded it");
            return crate::mtp::generate_mtp_spec_vulkan(&self.model, head, prompt, max_new, |p| {
                on_piece(p)
            });
        }
        self.ensure_session()?;
        self.model
            .generate_vulkan_session(self.session.as_mut().unwrap(), prompt, max_new, |p| {
                on_piece(p)
            })
    }

    fn generate_constrained(
        &mut self,
        prompt: &str,
        max_new: usize,
        constraint: &mut crate::grammar::Constraint,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.ensure_session()?;
        self.model.generate_vulkan_session_constrained(
            self.session.as_mut().unwrap(),
            prompt,
            max_new,
            Some(constraint),
            |p| on_piece(p),
        )
    }
}
