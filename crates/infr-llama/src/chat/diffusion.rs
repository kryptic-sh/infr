//! [`DiffusionGemmaChat`]: the diffusion-gemma [`ChatModel`], over a persistent per-backend
//! session ([`DiffusionSess`]) selected by [`DgBackend`] at construction.

use super::ChatModel;
use crate::{GenStats, SeamModel};
use anyhow::Result;

/// Either DiffusionGemma session (Phase 2/D, `seam/model.rs`) behind
/// [`crate::diffusion::DiffusionSession`] — lets [`DiffusionGemmaChat`] hold ONE persistent
/// session across turns regardless of backend.
enum DiffusionSess {
    // Boxed so the enum's size is a pointer regardless of which session it holds — the three
    // session structs differ enough in size to trip clippy's `large_enum_variant` (the
    // `#[cfg(macos)]` Metal variant makes the spread, so the check only fires on a macOS build —
    // the Linux clippy CI job never compiles it).
    Cpu(Box<crate::seam::model::DiffusionGemmaCpuSession>),
    Vulkan(Box<crate::seam::model::DiffusionGemmaVulkanSession>),
    /// Phase D: the Metal twin, macOS only (see `DiffusionGemmaChat::new_metal`).
    #[cfg(target_os = "macos")]
    Metal(Box<crate::seam::model::DiffusionGemmaMetalSession>),
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl crate::diffusion::DiffusionSession for DiffusionSess {
    fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()> {
        match self {
            DiffusionSess::Cpu(s) => s.prefill(model, tokens),
            DiffusionSess::Vulkan(s) => s.prefill(model, tokens),
            #[cfg(target_os = "macos")]
            DiffusionSess::Metal(s) => s.prefill(model, tokens),
        }
    }
    fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
        sample_temp_inv: f32,
        u: &[f32],
    ) -> Result<crate::seam::DenoiseOutcome> {
        use crate::seam::DenoiseOutcome;
        match self {
            DiffusionSess::Cpu(s) => Ok(DenoiseOutcome::Logits(s.denoise(
                model,
                canvas_tokens,
                sc_logits,
                temp_inv,
            )?)),
            DiffusionSess::Vulkan(s) => s.denoise(
                model,
                canvas_tokens,
                sc_logits,
                temp_inv,
                sample_temp_inv,
                Some(u),
            ),
            #[cfg(target_os = "macos")]
            DiffusionSess::Metal(s) => Ok(DenoiseOutcome::Logits(s.denoise(
                model,
                canvas_tokens,
                sc_logits,
                temp_inv,
            )?)),
        }
    }
}

/// Which backend [`DiffusionGemmaChat`] opens its session on — decided at construction, before a
/// session is ever opened (mirrors `DiffusionSess`'s three variants).
#[derive(Clone, Copy, PartialEq, Eq)]
enum DgBackend {
    Vulkan,
    Cpu,
    Metal,
}

/// diffusion-gemma (Phase 3/D — block text-diffusion, see `docs/diffusion-gemma.md` and
/// `crate::diffusion`): the entropy-bound decode loop over a persistent session, Vulkan by
/// default, or the CPU/Metal reference backends under `INFR_CPU`/`INFR_METAL` (Phase D added the
/// Metal DG session — macOS only; the non-macOS build still compiles `new_metal`, `generate`
/// errors clearly at runtime instead, matching every other INFR_METAL backend on this crate). The
/// session is opened lazily on the first turn (its KV cache is sized once the model's
/// `n_ctx_train`/`INFR_CTX` is known) and stays open across turns: multi-turn REPL re-sends
/// the WHOLE running token stream as the "prefix" each turn, and the session's own prefix-diff
/// prefill (see `DiffusionGemmaCpuSession::prefill`'s doc) re-sends only the un-cached suffix,
/// exactly like every other seam session on this crate.
pub struct DiffusionGemmaChat {
    model: SeamModel,
    backend: DgBackend,
    sess: Option<DiffusionSess>,
    max_ctx: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaChat {
    /// Production Vulkan session.
    pub fn new(model: SeamModel) -> Self {
        Self {
            model,
            backend: DgBackend::Vulkan,
            sess: None,
            max_ctx: 0,
        }
    }

    /// Reference CPU session (`INFR_CPU=1`).
    pub fn new_cpu(model: SeamModel) -> Self {
        Self {
            model,
            backend: DgBackend::Cpu,
            sess: None,
            max_ctx: 0,
        }
    }

    /// Reference Metal session (`INFR_METAL=1`, Phase D). Compiles on every target (like
    /// `CpuDenseChat::new_metal`) — the macOS check happens in `generate`, at session-open time.
    pub fn new_metal(model: SeamModel) -> Self {
        Self {
            model,
            backend: DgBackend::Metal,
            sess: None,
            max_ctx: 0,
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaChat {
    /// The turn plan for `prompt` tokens and a reply budget of `max_new` (see [`plan_turn`]):
    /// the session the turn needs and the budget the room left allows.
    fn plan(&self, prompt: usize, max_new: usize) -> Result<TurnPlan, crate::ContextOverflow> {
        let cfg = self.model.config();
        // `device.ctx` (INFR_CTX) shared size grammar; % resolves against the trained context (DG
        // sessions size against the canvas/prompt shape, not a VRAM-fit calc).
        let start = super::cfg_ctx(self.model.engine_cfg(), cfg.n_ctx_train)
            .unwrap_or_else(|| cfg.n_ctx_train.min(8192));
        plan_turn(prompt, max_new, cfg.canvas_length, start, cfg.n_ctx_train)
    }

    /// Open the session `plan` asks for, unless the open one already holds the turn — see the
    /// sizing comment at the call site in [`generate`](ChatModel::generate).
    fn ensure_sess(&mut self, plan: &TurnPlan) -> Result<()> {
        if self.sess.is_some() && plan.needed <= self.max_ctx {
            return Ok(());
        }
        let max_ctx = plan.session;
        self.max_ctx = max_ctx;
        self.sess = Some(match self.backend {
            DgBackend::Cpu => {
                DiffusionSess::Cpu(Box::new(self.model.diffusion_gemma_cpu_session(max_ctx)))
            }
            DgBackend::Vulkan => DiffusionSess::Vulkan(Box::new(
                self.model.diffusion_gemma_vulkan_session(max_ctx)?,
            )),
            DgBackend::Metal => {
                #[cfg(target_os = "macos")]
                {
                    DiffusionSess::Metal(Box::new(
                        self.model.diffusion_gemma_metal_session(max_ctx)?,
                    ))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    return Err(anyhow::anyhow!(
                        "the Metal backend is only available on macOS"
                    ));
                }
            }
        });
        Ok(())
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl ChatModel for DiffusionGemmaChat {
    fn render_model(&self) -> &SeamModel {
        &self.model
    }

    /// The block-diffusion decode has no grammar path.
    fn supports_constraint(&self) -> bool {
        false
    }

    /// The longest prompt one block still fits after: the trained context less one canvas and the
    /// session margin. A longer reply budget is trimmed to the room left, as the seam paths trim
    /// theirs.
    fn max_prompt_tokens(&self) -> Option<usize> {
        let cfg = self.model.config();
        Some(max_prompt(cfg.n_ctx_train, cfg.canvas_length))
    }

    /// Drops the whole session — its cached prefix along with its buffers — because the DG
    /// sessions have no token-only reset. The next turn reopens it lazily (`ensure_sess`)
    /// and re-prefills from scratch, paying the session open again.
    fn reset_kv(&mut self) {
        self.sess = None;
        self.max_ctx = 0;
    }

    fn warmup(&mut self) -> Result<()> {
        use crate::diffusion::DiffusionSession;
        // A tiny PREFILL-only forward compiles the lazily-built pipelines (GEMM/attention/batched
        // MoE — a cold DG prefill was measured at 26 t/s vs 1424 t/s warm, ~5s of one-time compile
        // otherwise billed to the first request). No denoise: the canvas plans build per (cc, p)
        // anyway, and their couple of extra pipelines (canvas attention, softmax) are cheap next
        // to the shared set warmed here. The throwaway tokens pollute the session's cached prefix
        // harmlessly — a real prompt prefix-diffs to 0 and re-prefills from scratch.
        crate::with_profiling_suppressed(|| -> Result<()> {
            let plan = self.plan(0, 0)?;
            self.ensure_sess(&plan)?;
            let enc = self
                .model
                .tokenizer()
                .encode("Hi", false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            self.sess
                .as_mut()
                .unwrap()
                .prefill(&self.model, enc.get_ids())
        })
    }

    /// `_req` is ACCEPTED AND IGNORED, which is exactly what this backend did before the
    /// per-sequence conversion too — and is therefore not a regression, but it IS a gap.
    ///
    /// DiffusionGemma does not sample in the autoregressive decode loop at all: its tokens come out
    /// of the block-diffusion denoise (`crate::diffusion::diffusion_generate`, `dg_eb_sample`),
    /// which has its OWN sampler and never consults `Sampler::resolve`. So the old thread-local
    /// never reached it either — a `temperature`/`seed`/`stop` on a DG serve request was already a
    /// no-op. Wiring per-request sampling (and the stop-sequence abort latch) into the denoise loop
    /// is a separate piece of work; it is reported, not silently papered over. The two request
    /// fields a client relies on to exclude or restrict tokens, `logit_bias` and `min_p`, are
    /// refused instead ([`crate::sampling::UnsupportedSampling`]): generating without them is not
    /// a reply the request asked for.
    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.generate_impl(prompt, max_new, req, on_piece, None)
    }

    fn generate_with_step_hook(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
        on_step: Option<&mut dyn FnMut(crate::diffusion::StepView)>,
    ) -> Result<GenStats> {
        self.generate_impl(prompt, max_new, req, on_piece, on_step)
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaChat {
    /// Shared body of [`ChatModel::generate`]/[`ChatModel::generate_with_step_hook`] — `on_step:
    /// None` (the plain `generate` path) is byte-identical to the pre-hook implementation: the
    /// hook is only ever read inside [`crate::diffusion::diffusion_generate`]'s step loop, and a
    /// `None` there is a single `if let` check with no other side effect.
    ///
    /// Streams each block's COMMITTED text through `on_piece` as soon as that block finishes
    /// (`diffusion_generate`'s `on_block`), rather than detokenizing the whole reply once
    /// `diffusion_generate` returns — this is what lets `INFR_DIFFUSION_VISUAL` print a finished
    /// block permanently while the next one is still denoising. The emitted BYTES are unchanged
    /// either way: `on_block` fires with exactly the same tokens, in the same order, that used to
    /// be detokenized in one pass over `result.tokens` afterward — only the timing/chunking of
    /// `on_piece` calls changes, never the concatenated text.
    fn generate_impl(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
        on_step: Option<&mut dyn FnMut(crate::diffusion::StepView)>,
    ) -> Result<GenStats> {
        // The denoise sampler reads no logit bias and has no min-p cut (see `generate`).
        if let Some(r) = req {
            r.sampling().refuse_seam_only_fields("diffusion-gemma")?;
        }
        let enc = self
            .model
            .tokenizer()
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();

        let cfg = self.model.config();
        let canvas_len = cfg.canvas_length;
        let vocab = cfg.vocab;
        let eos_ids = cfg.eos_ids.clone();
        let eb = crate::diffusion::EbConfig::from_config(cfg);
        // Seed determinism (see `crate::diffusion::diffusion_generate`'s doc): the reference
        // reseeds its RNG from a fixed value every block, so a fixed `sampling.seed` (INFR_SEED;
        // default 42 HERE — a two-defaults knob, both kept at their own site — matching the
        // oracle's `-s 42`) makes every turn reproducible.
        // The REQUEST's seed wins when it carries one. `GenParams.seed` is an accepted field that
        // `request_sampling` copies into `RequestSampling`, but this path used to read only the
        // process config, so two serve requests with different seeds produced identical output
        // (backlog B21). Falling back to `INFR_SEED`, then to 42 — a two-defaults knob, kept
        // at its own site, matching the oracle's `-s 42`.
        let seed: u64 = resolve_seed(
            req.and_then(|r| r.sampling().seed),
            self.model.engine_cfg().sampling.seed,
        );

        // Size the session to THIS turn's [prompt | every block's canvas] plus REPL headroom —
        // NOT n_ctx_train: DG's per-token KV is heavy (hd 256/512 across 30 layers ≈ 225 KB/tok)
        // and this model trains at 262144 ctx, so the n_ctx_train default every AR chat uses
        // would ask the backend for a ~59 GB KV cache (observed: radv device-lost at submit).
        // Default headroom is min(n_ctx_train, 8192) ≈ 1.8 GB — the same clamp the spec-decode
        // pair uses. A later REPL turn that outgrows the session reopens it bigger (the KV is
        // rebuilt by a from-scratch prefill; correct, just slower for that one turn).
        // A budget the context cannot hold is trimmed to the blocks that fit, as `reply_budget`
        // trims the seam paths'; only a prompt that leaves no room for one block is refused.
        let plan = self.plan(prompt_tokens.len(), max_new)?;
        self.ensure_sess(&plan)?;

        // Stream each block's committed tokens through the shared incremental UTF-8-safe detok —
        // the same helper every other backend's per-token decode loop uses (`crate::stream_token`)
        // — as soon as `diffusion_generate`'s `on_block` hands them over, so a block's text
        // appears as soon as it's decided rather than only after the whole turn finishes. `acc`/
        // `printed` persist across every block (not reset per block), same as they would across
        // every token of a single flat loop: a multi-byte char split across a block boundary is
        // held back exactly like one split across two tokens of the same block.
        let tokenizer = self.model.tokenizer();
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = crate::StreamOffsets::default();
        let mut on_block = |committed: &[u32]| {
            // The whole block's ids are in hand, so decode the committed span in ONE pass through
            // the shared incremental UTF-8-safe detok — NOT once per id over the growing `acc`
            // (that was O(total²): every token re-decoded the entire accumulator). Same holdback
            // semantics, same emitted bytes; only the per-token redundant decode is gone.
            crate::stream_tokens(
                tokenizer,
                &mut acc,
                &mut printed,
                committed,
                &mut |p: &str| on_piece(p),
            );
        };

        // The abort latch `run_chat` sets on a stop-sequence hit, a client disconnect, or the
        // request deadline firing. Polled at each block boundary — see `diffusion_generate`.
        let stop = req.map(|r| move || r.aborted());

        let result = crate::diffusion::diffusion_generate(
            self.sess.as_mut().unwrap(),
            &self.model,
            &prompt_tokens,
            canvas_len,
            vocab,
            &eos_ids,
            &eb,
            plan.max_new,
            seed,
            self.max_ctx,
            on_step,
            Some(&mut on_block),
            stop.as_ref().map(|f| f as &dyn Fn() -> bool),
        )?;

        let mut stats = result.stats;
        // A reply that used the whole of a budget the context trimmed ran out of context, which
        // serve reports as `length` (the seam paths say the same through `ctx_full`).
        stats.ctx_full |= plan.trimmed && stats.n_gen >= plan.max_new;
        Ok(stats)
    }
}

/// KV rows a session keeps past a turn's prompt and canvases, so a REPL's next turn usually fits
/// without reopening it.
const SESSION_MARGIN: usize = 64;

/// The longest prompt a turn can have in a trained context of `n_ctx_train`: one block's
/// `canvas` and [`SESSION_MARGIN`] must still fit after it.
fn max_prompt(n_ctx_train: usize, canvas: usize) -> usize {
    n_ctx_train.saturating_sub(canvas.max(1) + SESSION_MARGIN)
}

/// What one turn may use: see [`plan_turn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TurnPlan {
    /// KV rows the turn needs: the prompt, its blocks' canvases and [`SESSION_MARGIN`].
    needed: usize,
    /// The size to open a session at when the open one does not hold `needed`.
    session: usize,
    /// The reply budget after trimming.
    max_new: usize,
    /// Whether the budget was trimmed to the room left.
    trimmed: bool,
}

/// Plan a turn of `prompt` tokens with a reply budget of `max_new`, never past the trained
/// context `n_ctx_train`.
///
/// The budget is trimmed to the whole blocks of `canvas` tokens the context still holds after the
/// prompt — the seam paths' `reply_budget`, in blocks — so the prompt bound serve checks before
/// queueing ([`max_prompt`]) is the only way a turn is refused, and it is refused as the typed
/// overflow serve answers `400 context_length_exceeded`, quoting that same bound. The session
/// opens at `start` (`device.ctx`, else the default headroom), or larger when the turn needs it.
///
/// The cap is what stops an over-long prompt from opening an arbitrarily large session: DG's
/// per-token KV is heavy, and a session sized to whatever arrived could ask the backend for
/// gigabytes it does not have. There is no VRAM fit here: these sessions size against the canvas
/// and prompt shape, and the backend's allocation guard is the backstop below the cap.
fn plan_turn(
    prompt: usize,
    max_new: usize,
    canvas: usize,
    start: usize,
    n_ctx_train: usize,
) -> Result<TurnPlan, crate::ContextOverflow> {
    let canvas = canvas.max(1);
    let longest = max_prompt(n_ctx_train, canvas);
    if prompt > longest {
        // `max_ctx` is one past the longest prompt: the overflow message quotes `max_ctx - 1`,
        // the row the seam paths keep for the first reply token, which here is that bound.
        return Err(crate::ContextOverflow {
            prompt,
            gen: 0,
            max_ctx: longest + 1,
        });
    }
    let wanted = max_new.div_ceil(canvas).max(1);
    let fit = (n_ctx_train - prompt - SESSION_MARGIN) / canvas;
    let blocks = wanted.min(fit);
    let needed = prompt + blocks * canvas + SESSION_MARGIN;
    Ok(TurnPlan {
        needed,
        session: start.min(n_ctx_train).max(needed),
        max_new: max_new.min(blocks * canvas),
        trimmed: blocks < wanted,
    })
}

/// Which RNG seed a DiffusionGemma turn runs with: the REQUEST's if it carries one, else the
/// process config's (`INFR_SEED`), else 42.
///
/// Split out because the precedence is the whole of backlog B21's sampling half and the function
/// around it cannot be tested — `generate_impl` needs a loaded DiffusionGemma model. The bug was
/// that only the middle term was ever read, so two serve requests with different `seed` values
/// produced identical output.
///
/// The 42 is a two-defaults knob kept at its own site, and matches the reference CLI's
/// `-s 42`; DiffusionGemma reseeds per block, so a fixed seed makes a turn reproducible.
fn resolve_seed(req_seed: Option<u64>, cfg_seed: Option<u64>) -> u64 {
    req_seed.or(cfg_seed).unwrap_or(42)
}

#[cfg(test)]
mod tests {
    use super::{max_prompt, plan_turn, resolve_seed, SESSION_MARGIN};

    /// B82: a turn's session is sized to the turn but never past the trained context, and its
    /// reply budget is trimmed to the whole blocks that still fit — so the prompt bound serve
    /// checks before queueing is the only refusal, and the overflow quotes that bound.
    #[test]
    fn a_turn_is_trimmed_to_the_trained_context_and_refused_only_past_the_prompt_bound() {
        let (train, canvas, start) = (32_768, 256, 8_192);
        let plan = |prompt, max_new| plan_turn(prompt, max_new, canvas, start, train);
        let longest = max_prompt(train, canvas);
        assert_eq!(longest, train - canvas - SESSION_MARGIN);

        // A short turn: the default headroom, the budget untouched.
        let p = plan(100, 1_000).unwrap();
        assert_eq!((p.session, p.max_new, p.trimmed), (start, 1_000, false));
        assert_eq!(p.needed, 100 + 4 * canvas + SESSION_MARGIN);
        // A longer turn grows the session to exactly what it needs.
        let p = plan(20_000, 1_000).unwrap();
        assert_eq!(
            (p.session, p.trimmed),
            (20_000 + 4 * canvas + SESSION_MARGIN, false)
        );
        // A budget past the room left is trimmed to the blocks that fit, never past the context.
        let p = plan(31_000, 2_048).unwrap();
        let fit = (train - 31_000 - SESSION_MARGIN) / canvas;
        assert!(fit * canvas < 2_048);
        assert_eq!((p.max_new, p.trimmed), (fit * canvas, true));
        assert!(p.needed <= train && p.session <= train);
        // The longest prompt still gets its one block.
        let p = plan(longest, 2_048).unwrap();
        assert_eq!((p.max_new, p.needed), (canvas, train));
        // A configured start past the trained context is held to it.
        assert_eq!(
            plan_turn(100, 1, canvas, 65_536, train).unwrap().session,
            train
        );
        // One token past the bound is refused, and the message's limit is that bound.
        let err = plan(longest + 1, 1).unwrap_err();
        assert_eq!((err.prompt, err.gen), (longest + 1, 0));
        assert_eq!(err.max_ctx - 1, longest);
    }

    /// B21: a per-request seed must beat the process config, which used to be the only thing read.
    #[test]
    fn the_request_seed_wins_over_the_process_seed() {
        assert_eq!(resolve_seed(Some(7), Some(99)), 7, "request wins");
        assert_eq!(resolve_seed(Some(7), None), 7);
        // Two requests differing only in seed must not resolve to the same value — the reported
        // symptom was that they did.
        assert_ne!(
            resolve_seed(Some(7), Some(99)),
            resolve_seed(Some(8), Some(99))
        );
        // No request seed: fall back to the config, then to the oracle-matching default.
        assert_eq!(resolve_seed(None, Some(99)), 99);
        assert_eq!(resolve_seed(None, None), 42);
        // A request seed of 0 is a VALUE, not "unset".
        assert_eq!(resolve_seed(Some(0), Some(99)), 0);
    }
}
