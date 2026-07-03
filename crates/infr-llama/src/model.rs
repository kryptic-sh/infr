//! Agnostic chat orchestration: ONE shared [`Chat`] over a per-backend [`ChatModel`], so every
//! architecture runs the identical multi-turn REPL instead of 5 bespoke chat loops.
//!
//! Each backend keeps its own bespoke forward (dense `ChatSession` with a persistent KV cache, the
//! eager `generate_moe`, qwen35's per-token loop, the CPU compute-graph runner). A backend only
//! implements two primitives — [`render`](ChatModel::render) a message list through the model's OWN
//! chat template, and [`generate`](ChatModel::generate) a reply for an already-rendered prompt. The
//! shared [`Chat`] owns the conversation history, re-renders it every turn, strips the model's
//! `<think>` reasoning from what it stores, and reports [`GenStats`]. Turn orchestration lives in
//! ONE place; only the two primitives differ per architecture.
//!
//! The boxed trait object is lifetime-bounded (`Box<dyn ChatModel + '_>`) so the borrow-based dense
//! `ChatSession` (which borrows `&Llama`) needs no ownership change — the caller owns the `Llama`,
//! the box borrows it.

use crate::{no_template_err, CpuModel, GenStats};
use anyhow::Result;

/// The two arch-specific primitives the shared [`Chat`] drives. Object-safe: no generics, callbacks
/// are `&mut dyn FnMut`. A stateful backend (dense) may keep a KV cache warm across `generate` calls;
/// stateless ones prefill the whole prompt each turn.
pub trait ChatModel {
    /// Render a conversation `(role, content)` into a prompt string via the model's embedded chat
    /// template. All backends funnel to `infr_chat::render_chat_jinja` (the single prompt renderer).
    fn render(&self, messages: &[(&str, &str)]) -> Result<String>;

    /// Generate a completion for the already-rendered `prompt`, streaming decoded text to `on_piece`.
    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats>;

    /// Optional REPL status (e.g. dense returns `ctx N/MAX`); `None` for stateless backends.
    fn status(&self) -> Option<String> {
        None
    }

    /// Run a tiny throwaway generation through the real forward so every lazily-built pipeline
    /// compiles NOW instead of on the first user request (a cold Vulkan seam pays seconds of
    /// pipeline builds otherwise). INFR_PROF2 is suppressed for the duration — recorders read it
    /// at construction, and warmup submits would pollute a later bench's per-op aggregate.
    /// Default: no-op (stateless/CPU backends have nothing to warm).
    fn warmup(&mut self) -> Result<()> {
        Ok(())
    }

    /// Like [`generate`](Self::generate), with an llguidance grammar constraint applied to the
    /// decode (serve's FORCED tool_choice). Backends without constraint support return an error —
    /// the caller falls back to unconstrained generation.
    fn generate_constrained(
        &mut self,
        _prompt: &str,
        _max_new: usize,
        _constraint: &mut crate::grammar::Constraint,
        _on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        Err(anyhow::anyhow!(
            "grammar-constrained generation is not supported by this backend"
        ))
    }
}

/// Store only the ANSWER, dropping the model's reasoning (Qwen3 excludes prior-turn thinking;
/// keeping it degrades the model). Delegates to `infr-chat`'s splitter — the SAME reasoning
/// grammar `infr run`'s display and `infr serve`'s deltas use — so what history keeps always
/// matches what the surfaces call "content". Unterminated reasoning (truncated turn) → empty.
fn strip_think(reply: &str) -> String {
    infr_chat::split_reasoning(reply).1
}

/// The SINGLE arch-agnostic chat: owns the conversation history + `<think>`-stripping, drives any
/// [`ChatModel`] backend. Every architecture uses this one turn implementation and REPL, so a working
/// multi-turn REPL is uniform (no per-backend "one-shot only" special cases).
pub struct Chat<'a> {
    model: Box<dyn ChatModel + 'a>,
    history: Vec<(String, String)>,
}

impl<'a> Chat<'a> {
    /// Wrap a per-backend [`ChatModel`] in the shared multi-turn chat.
    pub fn new(model: Box<dyn ChatModel + 'a>) -> Self {
        Self {
            model,
            history: Vec::new(),
        }
    }

    /// Run one user turn: append the message, re-render the FULL history through the model's chat
    /// template, generate the reply (streamed to `on_piece`), strip the `<think>` reasoning from what
    /// we store, and keep the answer for the next turn. Returns per-turn [`GenStats`].
    pub fn turn(
        &mut self,
        message: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.history.push(("user".into(), message.to_string()));
        let prompt = {
            let refs: Vec<(&str, &str)> = self
                .history
                .iter()
                .map(|(r, c)| (r.as_str(), c.as_str()))
                .collect();
            match self.model.render(&refs) {
                Ok(p) => p,
                Err(e) => {
                    self.history.pop();
                    return Err(e);
                }
            }
        };
        let mut answer = String::new();
        let mut emit = |p: &str| {
            answer.push_str(p);
            on_piece(p);
        };
        // Template-prefilled thinking (Qwen3.5-style: the PROMPT ends with the `<think>` opener,
        // so the output starts mid-reasoning): inject a synthetic opener so the display styler
        // and the history stripper see a well-formed block.
        if infr_chat::prompt_prefills_think(&prompt) {
            emit("<think>");
        }
        let stats = self.model.generate(&prompt, max_new, &mut emit)?;
        self.history
            .push(("assistant".into(), strip_think(&answer)));
        Ok(stats)
    }

    /// Every backend now supports the interactive multi-turn REPL (kept for the CLI's call site).
    pub fn supports_repl(&self) -> bool {
        true
    }

    /// Optional status line (e.g. `ctx 12/8192`) for the REPL prompt.
    pub fn repl_status(&self) -> Option<String> {
        self.model.status()
    }
}

/// Which compute backend a [`Qwen35Chat`] loads its [`crate::qwen35::SeamModel`] on. All variants
/// run the SAME agnostic seam graph — only the executor differs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SeamBackend {
    /// Production GPU path (native-dtype weights in VRAM, in-kernel dequant).
    Vulkan,
    /// Reference CPU interpreter (`INFR_CPU=1`, zero-copy mmap weights).
    Cpu,
    /// Reference Apple-GPU backend (`INFR_METAL=1`, macOS only).
    Metal,
}

/// qwen35 / Qwen3-Next on the agnostic batched/chunked seam ([`crate::qwen35::SeamModel`]), loaded
/// ONCE on the first turn and reused after (weights stay resident across turns). One struct serves
/// every backend — Vulkan (production), CPU and Metal (reference) — and it is the same engine
/// `infr bench` times, so run and bench cannot drift apart.
pub struct Qwen35Chat {
    path: std::path::PathBuf,
    backend: SeamBackend,
    seam: Option<crate::qwen35::SeamModel>,
}

impl Qwen35Chat {
    /// Production Vulkan seam.
    pub fn new(path: std::path::PathBuf) -> Self {
        Self::with_backend(path, SeamBackend::Vulkan)
    }

    /// Reference CPU seam (`INFR_CPU=1`).
    pub fn new_cpu(path: std::path::PathBuf) -> Self {
        Self::with_backend(path, SeamBackend::Cpu)
    }

    /// Reference Metal seam (`INFR_METAL=1`).
    pub fn new_metal(path: std::path::PathBuf) -> Self {
        Self::with_backend(path, SeamBackend::Metal)
    }

    pub fn with_backend(path: std::path::PathBuf, backend: SeamBackend) -> Self {
        Self {
            path,
            backend,
            seam: None,
        }
    }
}

impl ChatModel for Qwen35Chat {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        crate::qwen35::render_chat_messages(&self.path, messages)
    }

    fn warmup(&mut self) -> Result<()> {
        let prof2 = std::env::var_os("INFR_PROF2");
        if prof2.is_some() {
            std::env::remove_var("INFR_PROF2");
        }
        // An undersized warmup SeamState is fine — a bigger real prompt rebuilds it (only the
        // compiled pipelines need to persist).
        let r = self.generate("Hi", 2, &mut |_| {});
        if let Some(v) = prof2 {
            std::env::set_var("INFR_PROF2", v);
        }
        r.map(|_| ())
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.seam.is_none() {
            self.seam = Some(match self.backend {
                SeamBackend::Vulkan => crate::qwen35::SeamModel::load_vulkan(&self.path)?,
                SeamBackend::Cpu => crate::qwen35::SeamModel::load_cpu(&self.path)?,
                SeamBackend::Metal => {
                    #[cfg(target_os = "macos")]
                    {
                        crate::qwen35::SeamModel::load_metal(&self.path)?
                    }
                    #[cfg(not(target_os = "macos"))]
                    return Err(anyhow::anyhow!(
                        "the Metal backend is only available on macOS"
                    ));
                }
            });
        }
        self.seam
            .as_mut()
            .unwrap()
            .generate(prompt, max_new, |p| on_piece(p))
    }

    fn generate_constrained(
        &mut self,
        prompt: &str,
        max_new: usize,
        constraint: &mut crate::grammar::Constraint,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.seam.is_none() {
            self.seam = Some(match self.backend {
                SeamBackend::Vulkan => crate::qwen35::SeamModel::load_vulkan(&self.path)?,
                SeamBackend::Cpu => crate::qwen35::SeamModel::load_cpu(&self.path)?,
                SeamBackend::Metal => {
                    #[cfg(target_os = "macos")]
                    {
                        crate::qwen35::SeamModel::load_metal(&self.path)?
                    }
                    #[cfg(not(target_os = "macos"))]
                    return Err(anyhow::anyhow!(
                        "the Metal backend is only available on macOS"
                    ));
                }
            });
        }
        self.seam
            .as_mut()
            .unwrap()
            .generate_constrained(prompt, max_new, Some(constraint), |p| on_piece(p))
    }
}

/// Standalone OpenAI-shaped prompt renderer over a GGUF's own chat template — tool specs and prior
/// tool calls/results included (`infr_chat::render_chat_oai`). Model-independent: the serve-side
/// render for seam-backed [`ChatModel`]s, so serve never needs the bespoke `Llama` just to render.
pub struct OaiRenderer {
    gguf: infr_gguf::Gguf,
    tokenizer: tokenizers::Tokenizer,
    eos: u32,
}

impl OaiRenderer {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let gguf = infr_gguf::Gguf::open(path).map_err(|e| anyhow::anyhow!("open gguf: {e}"))?;
        let tokenizer = crate::build_tokenizer(&gguf)?;
        // Raw metadata (NOT Config::from_gguf — that parser is dense-only and rejects qwen35).
        use infr_core::WeightSource;
        let eos = gguf
            .metadata()
            .u64("tokenizer.ggml.eos_token_id")
            .unwrap_or(2) as u32;
        Ok(Self {
            gguf,
            tokenizer,
            eos,
        })
    }

    pub fn render(
        &self,
        messages: &[infr_chat::ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        infr_chat::render_chat_oai(&self.gguf, &self.tokenizer, self.eos, messages, tools, true)
            .ok_or_else(no_template_err)
    }

    /// Build the FORCED tool-call grammar constraint for this model's tokenizer (see
    /// [`crate::grammar::tool_constraint_for`]); `None` for auto/none/absent tool_choice.
    pub fn tool_constraint(
        &self,
        tools: Option<&serde_json::Value>,
        tool_choice: Option<&str>,
    ) -> Result<Option<crate::grammar::Constraint>> {
        let vocab = self.tokenizer.get_vocab_size(true);
        crate::grammar::tool_constraint_for(&self.tokenizer, vocab, &[self.eos], tools, tool_choice)
    }

    /// Detokenize ids (markers preserved) — the serve-side parse of a constrained tool-call body.
    pub fn decode_ids(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}

/// Dense/MoE on the VULKAN agnostic seam with a persistent KV session (`INFR_SEAM=1` for
/// `infr run`): weights upload once, and every turn prefills only the token suffix that differs
/// from the previous turn — the seam twin of the bespoke `ChatSession`'s incremental prefill.
pub struct DenseSeamChat {
    model: CpuModel,
    session: Option<crate::cpu_model::DenseVulkanSession>,
}

impl DenseSeamChat {
    pub fn new(model: CpuModel) -> Self {
        Self {
            model,
            session: None,
        }
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
        if self.session.is_none() {
            let max_ctx = std::env::var("INFR_MAX_CTX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(self.model.config().n_ctx_train);
            self.session = Some(self.model.vulkan_session(max_ctx)?);
        }
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
        if self.session.is_none() {
            let max_ctx = std::env::var("INFR_MAX_CTX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(self.model.config().n_ctx_train);
            self.session = Some(self.model.vulkan_session(max_ctx)?);
        }
        self.model.generate_vulkan_session_constrained(
            self.session.as_mut().unwrap(),
            prompt,
            max_new,
            Some(constraint),
            |p| on_piece(p),
        )
    }
}

/// Metal seam backend for dense/MoE with a persistent session — the Apple-GPU twin of
/// [`DenseSeamChat`]: weights upload once, the KV cache persists across turns, and each turn
/// prefills only the suffix that differs from the previous rendered history.
#[cfg(target_os = "macos")]
pub struct MetalSeamChat {
    model: CpuModel,
    session: Option<crate::cpu_model::DenseMetalSession>,
}

#[cfg(target_os = "macos")]
impl MetalSeamChat {
    pub fn new(model: CpuModel) -> Self {
        Self {
            model,
            session: None,
        }
    }

    fn ensure_session(&mut self) -> Result<()> {
        if self.session.is_none() {
            let max_ctx = std::env::var("INFR_MAX_CTX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(self.model.config().n_ctx_train);
            self.session = Some(self.model.metal_session(max_ctx)?);
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl ChatModel for MetalSeamChat {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.model.render_chat_messages(messages)
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.ensure_session()?;
        self.model
            .generate_metal_session(self.session.as_mut().unwrap(), prompt, max_new, |p| {
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
        self.model.generate_metal_session_constrained(
            self.session.as_mut().unwrap(),
            prompt,
            max_new,
            Some(constraint),
            |p| on_piece(p),
        )
    }
}

/// Speculative Metal chat (`INFR_SPEC_DRAFT=<gguf>`): TARGET model verified decode with a small
/// same-tokenizer DRAFT proposing k tokens per round. Greedy-only — the committed stream is
/// exactly the target's greedy stream. Pays off for ≥8B-class targets (see issue #16's
/// measurements); the CLI warns when the size ratio looks too thin.
#[cfg(target_os = "macos")]
pub struct SpecMetalChat {
    target: CpuModel,
    draft: CpuModel,
    k: usize,
    target_session: Option<crate::cpu_model::DenseMetalSession>,
    draft_session: Option<crate::cpu_model::DenseMetalSession>,
}

#[cfg(target_os = "macos")]
impl SpecMetalChat {
    pub fn new(target: CpuModel, draft: CpuModel, k: usize) -> Self {
        Self {
            target,
            draft,
            k,
            target_session: None,
            draft_session: None,
        }
    }

    fn ensure_sessions(&mut self) -> Result<()> {
        if self.target_session.is_none() {
            // TWO models + TWO KV caches share the working set — a full-n_ctx_train pair
            // (40k tokens on qwen3) thrashes an 18 GB machine into second-long forwards.
            // Default to 8k unless INFR_MAX_CTX says otherwise.
            let max_ctx = std::env::var("INFR_MAX_CTX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| self.target.config().n_ctx_train.min(8192));
            self.target_session = Some(self.target.metal_session(max_ctx)?);
            self.draft_session = Some(self.draft.metal_session(max_ctx)?);
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl ChatModel for SpecMetalChat {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.target.render_chat_messages(messages)
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.ensure_sessions()?;
        // Split borrows: sessions come out of the options for the call.
        let mut ts = self.target_session.take().unwrap();
        let mut ds = self.draft_session.take().unwrap();
        let r = self.target.generate_metal_spec(
            &mut ts,
            &self.draft,
            &mut ds,
            prompt,
            max_new,
            self.k,
            |p| on_piece(p),
        );
        self.target_session = Some(ts);
        self.draft_session = Some(ds);
        r
    }
}

/// CPU reference backend (`INFR_CPU=1`) for dense/MoE: the agnostic compute-graph forward, no GPU.
/// Stateless full-prefill each turn (no cross-turn KV yet), but the shared `Chat` now feeds the FULL
/// rendered history in every turn, so multi-turn context works.
pub struct CpuDenseChat {
    model: CpuModel,
    /// Run the dense forward on the reference Metal backend instead of the CPU interpreter.
    metal: bool,
}

impl CpuDenseChat {
    pub fn new(model: CpuModel) -> Self {
        Self {
            model,
            metal: false,
        }
    }

    /// Same dense model, but driven through the reference Metal backend (`INFR_METAL`).
    pub fn new_metal(model: CpuModel) -> Self {
        Self { model, metal: true }
    }
}

impl ChatModel for CpuDenseChat {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.model.render_chat_messages(messages)
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.metal {
            #[cfg(target_os = "macos")]
            return self.model.generate_metal(prompt, max_new, |p| on_piece(p));
            #[cfg(not(target_os = "macos"))]
            return Err(anyhow::anyhow!(
                "the Metal backend is only available on macOS"
            ));
        }
        self.model.generate_cpu(prompt, max_new, |p| on_piece(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scripted [`ChatModel`] that records every rendered prompt and replies with a canned string
    /// (optionally wrapped in `<think>`). Proves the shared [`Chat`] orchestration WITHOUT a model:
    /// history accumulates, both prior turns render into turn 2's prompt, and `<think>` is stripped
    /// from stored history but streamed to the caller.
    struct ScriptModel {
        rendered: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
        reply: String,
    }

    impl ChatModel for ScriptModel {
        fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
            // A trivial deterministic "template": one line per message.
            let s = messages
                .iter()
                .map(|(r, c)| format!("{r}: {c}"))
                .collect::<Vec<_>>()
                .join("\n");
            self.rendered.borrow_mut().push(s.clone());
            Ok(s)
        }

        fn generate(
            &mut self,
            _prompt: &str,
            _max_new: usize,
            on_piece: &mut dyn FnMut(&str),
        ) -> Result<GenStats> {
            on_piece(&self.reply);
            Ok(GenStats::default())
        }
    }

    #[test]
    fn shared_chat_accumulates_history_and_strips_think() {
        let rendered = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let model = ScriptModel {
            rendered: rendered.clone(),
            reply: "<think>reason</think>\n\nParis".into(),
        };
        let mut chat = Chat::new(Box::new(model));

        // Turn 1.
        let mut streamed = String::new();
        chat.turn("capital of France?", 8, &mut |p| streamed.push_str(p))
            .unwrap();
        // The caller sees the full reply (think included — the CLI's ThinkRender dims it live).
        assert_eq!(streamed, "<think>reason</think>\n\nParis");
        // Stored history keeps only the answer after </think>.
        assert_eq!(
            chat.history[0],
            ("user".into(), "capital of France?".into())
        );
        assert_eq!(chat.history[1], ("assistant".into(), "Paris".into()));

        // Turn 2: the rendered prompt must include BOTH prior turns (multi-turn context) with the
        // think-free assistant answer, then the new user message.
        chat.turn("and of Italy?", 8, &mut |_| {}).unwrap();
        let second = &rendered.borrow()[1];
        assert!(second.contains("user: capital of France?"), "{second}");
        assert!(second.contains("assistant: Paris"), "{second}");
        assert!(
            !second.contains("reason"),
            "prior think must be excluded: {second}"
        );
        assert!(second.contains("user: and of Italy?"), "{second}");
    }

    #[test]
    fn strip_think_variants() {
        // Reasoning spans stripped (ALL of them), content concatenated and trimmed.
        assert_eq!(strip_think("<think>a</think>b"), "b");
        assert_eq!(strip_think("<think>a</think>x<think>c</think>y"), "xy");
        // Unterminated think → nothing.
        assert_eq!(strip_think("<think>a"), "");
        // No think → the whole reply.
        assert_eq!(strip_think("plain"), "plain");
    }
}
