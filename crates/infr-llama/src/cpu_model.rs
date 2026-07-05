//! The GPU-free `CpuModel` for the CPU reference backend (no Vulkan/VRAM; weights streamed
//! from the GGUF mmap at forward time). Split out of `lib.rs` (no logic change).
use crate::*;
use anyhow::{anyhow, Result};
use infr_chat::{render_chat_jinja, render_chat_user};
use infr_core::backend::{Backend, BufferUsage};
use infr_cpu::CpuBackend;
use infr_gguf::Gguf;
use std::path::Path;
use tokenizers::Tokenizer;

/// A **GPU-free** model for the CPU reference backend. Holds only what the agnostic CPU compute
/// graph needs — the parsed [`Config`], the host f32 token embeddings (for the gather + tied lm
/// head), the tokenizer, and the gemma4 E2B per-layer-embd tensors. No `VulkanBackend`, no VRAM,
/// no weight upload: the projection weights are streamed straight from the kept-open GGUF mmap at
/// forward time. Dense Qwen3/Llama, Gemma 3, Gemma 4 (dense + E2B), and qwen3moe; for qwen35 use
/// [`crate::qwen35::generate_cpu`].
pub struct CpuModel {
    gguf: Gguf,
    cfg: Config,
    token_embd: Vec<f32>,
    per_layer_embd: Option<PerLayerEmbd>,
    tokenizer: Tokenizer,
}

/// The conversation SLOTS a persistent GPU seam session owns: up to `INFR_KV_SLOTS` (default 4)
/// [`crate::cpu_backend::SeamKv`]s — each a KV cache + the token ids materialized in it, all
/// sharing one weight upload (`Arc<SeamWeights>`). Per request the best-prefix slot is picked: a
/// prompt that EXTENDS a slot's cache continues it (the classic next-turn suffix prefill); a
/// prompt that diverges early (a different conversation) forks a fresh slot and SEEDS it with the
/// longest shared prefix (e.g. a common system prompt) via a device-side KV copy instead of
/// re-prefilling it; when all slots are taken the LRU one is recycled. Single-conversation
/// callers (run/bench/spec drivers) stay on one slot. Backend-agnostic: fork and seed go through
/// `&dyn Backend` (`copy_buffer` is the seeding primitive), so the Vulkan and Metal sessions
/// share this policy verbatim.
struct SlotPool {
    slots: Vec<Option<crate::cpu_backend::SeamKv>>,
    last_used: Vec<u64>,
    tick: u64,
}

/// A persistent Vulkan seam session (see [`CpuModel::vulkan_session`]): owns the backend and the
/// conversation [`SlotPool`].
pub struct DenseVulkanSession {
    vk: infr_vulkan::VulkanBackend,
    pool: SlotPool,
    max_ctx: usize,
}

impl DenseVulkanSession {
    /// Forget every slot's materialized tokens (buffers and the weight upload stay) — discards a
    /// warmup generation so the first real prompt starts from clean slots.
    pub fn reset_cache(&mut self) {
        self.pool.reset_cache();
    }
}

impl SlotPool {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            last_used: Vec::new(),
            tick: 0,
        }
    }

    /// Forget every slot's materialized tokens (buffers and the weight upload stay) — discards a
    /// warmup generation so the first real prompt starts from clean slots.
    fn reset_cache(&mut self) {
        for s in self.slots.iter_mut().flatten() {
            s.reset();
        }
    }

    /// The single-conversation slot (the bench/spec drivers, which manage one token stream and
    /// never contend): slot 0, created empty on first use.
    #[cfg(target_os = "macos")]
    fn single(&mut self) -> &mut Option<crate::cpu_backend::SeamKv> {
        if self.slots.is_empty() {
            self.slots.push(None);
            self.last_used.push(self.tick);
        }
        &mut self.slots[0]
    }

    /// Pick (and prepare) the slot for `prompt`; returns its index. See the struct doc for the
    /// policy. A freshly created slot is `None` — the runner's first call uploads the weights.
    ///
    /// This best-prefix choice is prefix-OPTIMAL for a real per-position KV cache (dense/
    /// attention arches): the picked slot always has the longest reusable prefix. For qwen35
    /// (gated-DeltaNet: an append-only recurrent summary, not a per-position cache — see the
    /// no-rewind rule in `cpu_backend::generate_dense_backend`) a `prefix_score` match is only
    /// scored on shared TOKENS, not on whether the state can actually rewind to it — so the pick
    /// here is merely CORRECT, not necessarily optimal: the runner independently re-checks EXACT
    /// extension (`prompt` extends `cached` verbatim) before reusing the slot's state, and
    /// zero-resets it otherwise. A suboptimal pick for qwen35 just costs an extra full
    /// re-prefill; it can never reuse a wrong recurrent state.
    fn pick(
        &mut self,
        be: &dyn infr_core::backend::Backend,
        cfg: &crate::Config,
        prompt: &[u32],
    ) -> Result<usize> {
        // Seeding shorter prefixes than this isn't worth the copy submit.
        const MIN_SEED: usize = 16;
        let max_slots: usize = std::env::var("INFR_KV_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| v > 0)
            .unwrap_or(4);
        self.tick += 1;
        if self.slots.is_empty() {
            self.slots.push(None);
            self.last_used.push(self.tick);
            return Ok(0);
        }
        let score = |st: &Option<crate::cpu_backend::SeamKv>| {
            st.as_ref().map_or(0, |s| s.prefix_score(prompt))
        };
        // A slot whose cache the prompt EXTENDS (or equals) is this conversation continuing.
        if let Some(i) = (0..self.slots.len()).find(|&i| {
            self.slots[i].as_ref().is_some_and(|s| {
                let p = s.prefix_score(prompt);
                p > 0 && (p == s.cached_len() || p == prompt.len())
            })
        }) {
            self.last_used[i] = self.tick;
            return Ok(i);
        }
        // Different conversation: the best shared prefix (if any) seeds the slot we hand out.
        let (best_i, best_s) = (0..self.slots.len())
            .map(|i| (i, score(&self.slots[i])))
            .max_by_key(|&(_, s)| s)
            .unwrap();
        let target = if self.slots.len() < max_slots {
            // Fork a fresh slot off any initialized one (shared weights, own KV).
            let src = self
                .slots
                .iter()
                .flatten()
                .next()
                .expect("pick_slot: no initialized slot to fork from");
            let fresh = src.fork(be, cfg)?;
            self.slots.push(Some(fresh));
            self.last_used.push(self.tick);
            self.slots.len() - 1
        } else {
            // Recycle the least-recently-used slot.
            (0..self.slots.len())
                .min_by_key(|&i| self.last_used[i])
                .unwrap()
        };
        if best_s >= MIN_SEED && best_i != target {
            // Give the slot the shared prefix (system prompt etc.) via device-side KV copy —
            // only when it beats whatever prefix the slot already shares with the prompt.
            if best_s > score(&self.slots[target]) {
                let src = self.slots[best_i].take().expect("scored slot is Some");
                if let Some(dst) = self.slots[target].as_mut() {
                    dst.seed_from(be, cfg, &src, best_s)?;
                }
                self.slots[best_i] = Some(src);
            }
        }
        self.last_used[target] = self.tick;
        Ok(target)
    }
}

/// A persistent Metal seam session — the Apple-GPU twin of [`DenseVulkanSession`]: owns the
/// backend and the conversation [`SlotPool`], so every later
/// [`CpuModel::generate_metal_session`] call prefills only the suffix that differs from its
/// slot's previous turn, and concurrent conversations (serve) each keep their own KV slot off
/// the one shared weight upload. Slot switches re-record the decode replay tape (its fingerprint
/// covers the bound KV/IO buffer addresses) — one graph-walk token per switch, never a stale
/// replay.
#[cfg(target_os = "macos")]
pub struct DenseMetalSession {
    mtl: infr_metal::MetalBackend,
    pool: SlotPool,
    max_ctx: usize,
}

#[cfg(target_os = "macos")]
impl DenseMetalSession {
    /// Forget every slot's materialized tokens (buffers and the weight upload stay) — discards a
    /// warmup generation so the first real prompt starts from clean slots.
    pub fn reset_cache(&mut self) {
        self.pool.reset_cache();
    }
}

impl CpuModel {
    /// Load a model for CPU inference without touching the GPU. `tokenizer_path` overrides the
    /// GGUF's embedded vocab when given.
    pub fn load(gguf_path: &Path, tokenizer_path: Option<&Path>) -> Result<Self> {
        let g = Gguf::open(gguf_path).map_err(|e| anyhow!("open gguf: {e}"))?;
        let mut cfg = Config::from_gguf(&g)?;
        let tokenizer = match tokenizer_path {
            Some(p) => Tokenizer::from_file(p).map_err(|e| anyhow!("load tokenizer: {e}"))?,
            None => build_tokenizer(&g)?,
        };
        add_chat_eos(&mut cfg, &tokenizer);
        let (token_embd, _) = load_tensor_dequant(&g, "token_embd.weight")?;
        let per_layer_embd = build_per_layer_embd(&g, &cfg)?;
        Ok(Self {
            gguf: g,
            cfg,
            token_embd,
            per_layer_embd,
            tokenizer,
        })
    }

    /// Open a persistent Vulkan seam session: weights uploaded ONCE, the KV cache sized to
    /// `max_ctx`, and the materialized-token cache that makes every later
    /// [`generate_vulkan_session`](Self::generate_vulkan_session) call prefill only the suffix
    /// that differs from the previous turn (ChatSession-style KV reuse on the agnostic seam).
    pub fn vulkan_session(&self, max_ctx: usize) -> Result<DenseVulkanSession> {
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        Ok(DenseVulkanSession {
            vk,
            pool: SlotPool::new(),
            max_ctx,
        })
    }

    /// Greedy generation on the Vulkan seam through a persistent session (see
    /// [`vulkan_session`](Self::vulkan_session)). `stats.n_prompt` reports the tokens actually
    /// PREFILLED (the un-cached suffix) — the TTFT-honest count.
    pub fn generate_vulkan_session(
        &self,
        session: &mut DenseVulkanSession,
        prompt: &str,
        max_new: usize,
        on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        self.generate_vulkan_session_constrained(session, prompt, max_new, None, on_piece)
    }

    /// [`generate_vulkan_session`](Self::generate_vulkan_session) with an optional llguidance
    /// grammar constraint (serve's forced tool_choice) applied to the decode.
    pub fn generate_vulkan_session_constrained(
        &self,
        session: &mut DenseVulkanSession,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let slot = session.pool.pick(&session.vk, &self.cfg, &prompt_tokens)?;
        let max_ctx = session.max_ctx;
        let (_generated, stats) = crate::cpu_backend::generate_dense_gpu_session(
            &session.vk,
            &self.gguf,
            &self.cfg,
            &self.token_embd,
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
            &mut session.pool.slots[slot],
            max_ctx,
            constraint,
        )?;
        Ok(stats)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Tokenize raw text with the model's own tokenizer (no chat template) — for callers that
    /// need token ids directly (e.g. a raw-forward validation harness), as opposed to
    /// [`render_chat`](Self::render_chat) + the generation loop.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Raw tokenizer accessor for callers outside this module that need incremental detok (the
    /// diffusion decode loop, Phase 3) via the shared [`crate::stream_token`] helper.
    pub(crate) fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Detokenize ids back to text (`encode`'s twin, `skip_special_tokens=true` — matches
    /// [`crate::stream_token`]'s convention so a thinking model's `<|channel>thought`/`<channel|>`
    /// markers, which aren't in the tokenizer's added-specials set, still come through as text) —
    /// for callers that drive a decode loop directly on token ids (the diffusion decode loop's own
    /// tests) instead of through one of the `generate_*` string-in/string-out helpers.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// DiffusionGemma Phase-1 validation: a causal prefill of `tokens` on the CPU reference
    /// backend, returning the LAST token's raw logits (`[vocab]`, pre-softmax, post-softcap). Not
    /// specific to diffusion-gemma — works for any arch on this seam — but this is its only
    /// caller today (see `docs/DIFFUSIONGEMMA.md`).
    pub fn prefill_logits_cpu(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        crate::cpu_backend::verify_dense_cpu(
            &self.gguf,
            &self.cfg,
            &self.token_embd,
            self.per_layer_embd.as_ref(),
            tokens,
        )
    }

    /// [`prefill_logits_cpu`](Self::prefill_logits_cpu)'s Vulkan twin, for the CPU/Vulkan
    /// cross-backend parity check.
    pub fn prefill_logits_vulkan(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        crate::cpu_backend::verify_dense_vulkan(
            &vk,
            &self.gguf,
            &self.cfg,
            &self.token_embd,
            self.per_layer_embd.as_ref(),
            tokens,
        )
    }

    /// Open a Phase-2 DiffusionGemma denoise session on the CPU reference backend (see
    /// `docs/DIFFUSIONGEMMA.md`): [`prefill`](DiffusionGemmaCpuSession::prefill) causally
    /// prefills the prompt ONCE (encoder scalars, KV rows `0..P`), then repeated
    /// [`denoise`](DiffusionGemmaCpuSession::denoise) calls forward the C-row canvas (decoder
    /// scalars, the bidirectional `Canvas` mask) against the SAME session — each call OVERWRITES
    /// KV rows `P..P+C`, so a caller can denoise the same block with a different partially-
    /// unmasked canvas over and over (the loop Phase 3 drives). `max_ctx` sizes the session's KV
    /// cache (must fit the whole prompt + canvas_length + any headroom for later blocks).
    pub fn diffusion_gemma_cpu_session(&self, max_ctx: usize) -> DiffusionGemmaCpuSession {
        DiffusionGemmaCpuSession {
            be: CpuBackend::new(),
            state: None,
            max_ctx,
        }
    }

    /// [`diffusion_gemma_cpu_session`](Self::diffusion_gemma_cpu_session)'s Vulkan twin, for the
    /// CPU/Vulkan cross-backend parity check.
    pub fn diffusion_gemma_vulkan_session(
        &self,
        max_ctx: usize,
    ) -> Result<DiffusionGemmaVulkanSession> {
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        Ok(DiffusionGemmaVulkanSession {
            be: vk,
            state: None,
            max_ctx,
        })
    }

    /// Token-level bench on the CPU reference backend (no GPU): prefill `n_prompt` dummy tokens, then
    /// decode `n_gen`, returning the timing ([`crate::GenStats`] has `prompt_secs`/`decode_secs`). Lets
    /// `infr bench -ngl 0` measure prefill (pp = n_prompt/prompt_secs) and decode (tg = n_gen/decode_secs)
    /// directly comparable to `llama-bench -ngl 0`. Dummy tokens — timing is data-independent.
    pub fn bench(&self, n_prompt: usize, n_gen: usize) -> Result<crate::GenStats> {
        let prompt: Vec<u32> = (0..n_prompt.max(1)).map(|i| (i % 100) as u32).collect();
        let (_, stats) = crate::cpu_backend::generate_dense_cpu(
            &self.gguf,
            &self.cfg,
            &self.token_embd,
            self.per_layer_embd.as_ref(),
            &prompt,
            n_gen,
            |_| {},
        )?;
        Ok(stats)
    }

    /// Run the dense decode through the agnostic compute seam on the **Vulkan** backend — the GPU
    /// twin of [`generate_cpu`](Self::generate_cpu). Each native-dtype GGUF weight is padded + uploaded
    /// to VRAM (the CPU path maps it zero-copy instead); the per-token [`infr_core::graph::Graph`] is
    /// compiled + executed by `VulkanBackend`; greedy tokens are detokenized. Same graph, two
    /// backends — this is the end-to-end dense CPU↔GPU parity path.
    pub fn generate_dense_vulkan(&self, prompt: &str, max_new: usize) -> Result<String> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        let (generated, _stats) = crate::cpu_backend::generate_dense_gpu(
            &vk,
            &self.gguf,
            &self.cfg,
            &self.token_embd,
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |_| {},
        )?;
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Token-level bench on the Vulkan seam, llama-bench-comparable: ONE weight upload (a
    /// persistent session) + an untimed pipeline warmup, then per rep — reset the KV, warm it to
    /// `depth` (untimed), and time ONE metric: `pg` = a whole (P prefill + G decode) turn,
    /// `n_gen > 0` = decode at depth, else prefill of `n_prompt` at depth. Returns the per-rep
    /// tokens/sec samples.
    pub fn bench_vulkan(
        &self,
        n_prompt: usize,
        n_gen: usize,
        depth: usize,
        pg: Option<(usize, usize)>,
        reps: usize,
    ) -> Result<Vec<f64>> {
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        let (p_eff, g_eff) = pg.unwrap_or((n_prompt, n_gen));
        // +16: the untimed warmup runs an 8-prompt + 2-gen turn through the SAME session, so the
        // KV must fit it even when the measured shape is tiny (pp2 with +8 sized the cache to 10
        // and the warmup itself overflowed it).
        let want = depth + p_eff.max(1) + g_eff + 16;
        let dummy = |n: usize| -> Vec<u32> { (0..n.max(1)).map(|i| (i % 100) as u32).collect() };
        let mut state: Option<crate::cpu_backend::SeamKv> = None;
        let run = |prompt_len: usize,
                   gen: usize,
                   state: &mut Option<crate::cpu_backend::SeamKv>|
         -> Result<crate::GenStats> {
            let (_, stats) = crate::cpu_backend::generate_dense_gpu_session(
                &vk,
                &self.gguf,
                &self.cfg,
                &self.token_embd,
                self.per_layer_embd.as_ref(),
                &dummy(prompt_len),
                gen,
                |_| {},
                state,
                want,
                None,
            )?;
            Ok(stats)
        };
        // Untimed warmup: uploads the weights and compiles every pipeline the timed reps hit.
        run(8, 2, &mut state)?;
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps.max(1) {
            if let Some(st) = state.as_mut() {
                st.reset();
            }
            if depth > 0 {
                run(depth, 0, &mut state)?; // warm the cache to `depth` (untimed)
            }
            if let Some((p, g)) = pg {
                // coding-agent turn: prompt ingest + reply generation timed together.
                let s = run(depth + p, g, &mut state)?;
                samples.push((p + g) as f64 / (s.prompt_secs + s.decode_secs).max(1e-9));
            } else if n_gen > 0 {
                // decode at depth: 1-token suffix feeds the loop, the timed part is the decode.
                let s = run(depth + 1, n_gen, &mut state)?;
                samples.push(n_gen as f64 / s.decode_secs.max(1e-9));
            } else {
                // +1: the suffix's LAST token is the decode feed and is never processed at
                // gen=0, and a suffix of <= 2 skips batched prefill entirely — so `depth + N`
                // measured N-1 batched rows (and pp2 measured nothing, reporting the 1e-9
                // floor). With +1, exactly N rows batch-prefill (positions depth..depth+N) and
                // prompt_secs covers precisely them — llama-bench's -p N semantics.
                let s = run(depth + n_prompt + 1, 0, &mut state)?;
                samples.push(n_prompt as f64 / s.prompt_secs.max(1e-9));
            }
        }
        Ok(samples)
    }

    /// Open a persistent Metal seam session (the Apple-GPU twin of
    /// [`vulkan_session`](Self::vulkan_session)): weights uploaded ONCE, KV sized to `max_ctx`,
    /// later calls prefill only the un-cached suffix.
    #[cfg(target_os = "macos")]
    pub fn metal_session(&self, max_ctx: usize) -> Result<DenseMetalSession> {
        let mtl = infr_metal::MetalBackend::new().map_err(|e| anyhow!("metal init: {e}"))?;
        Ok(DenseMetalSession {
            mtl,
            pool: SlotPool::new(),
            max_ctx,
        })
    }

    /// Greedy generation on the Metal seam through a persistent session (see
    /// [`metal_session`](Self::metal_session)). `stats.n_prompt` reports the tokens actually
    /// PREFILLED (the un-cached suffix) — the TTFT-honest count.
    #[cfg(target_os = "macos")]
    pub fn generate_metal_session(
        &self,
        session: &mut DenseMetalSession,
        prompt: &str,
        max_new: usize,
        on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        self.generate_metal_session_constrained(session, prompt, max_new, None, on_piece)
    }

    /// [`generate_metal_session`](Self::generate_metal_session) with an optional llguidance
    /// grammar constraint (serve's forced tool_choice) applied to the decode.
    #[cfg(target_os = "macos")]
    pub fn generate_metal_session_constrained(
        &self,
        session: &mut DenseMetalSession,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let slot = session.pool.pick(&session.mtl, &self.cfg, &prompt_tokens)?;
        let (_generated, stats) = crate::cpu_backend::generate_dense_metal_session(
            &session.mtl,
            &self.gguf,
            &self.cfg,
            &self.token_embd,
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
            &mut session.pool.slots[slot],
            session.max_ctx,
            constraint,
        )?;
        Ok(stats)
    }

    /// Greedy generation on the reference **Metal** backend through the agnostic seam (the
    /// Apple-GPU twin of [`generate_cpu`](Self::generate_cpu)): weights are uploaded to Metal
    /// buffers, the per-token [`infr_core::graph::Graph`] is compiled + executed by `MetalBackend`,
    /// and generated tokens stream through `on_piece`.
    #[cfg(target_os = "macos")]
    pub fn generate_metal(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let mtl = infr_metal::MetalBackend::new().map_err(|e| anyhow!("metal init: {e}"))?;
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (_generated, stats) = crate::cpu_backend::generate_dense_metal(
            &mtl,
            &self.gguf,
            &self.cfg,
            &self.token_embd,
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
        )?;
        Ok(stats)
    }

    /// Speculative decoding on the Metal seam (`self` = TARGET, `draft` = the small
    /// same-tokenizer model): the draft session proposes `k` greedy tokens, ONE batched verify
    /// forward of the target checks all of them (LM head on every candidate row), the matching
    /// prefix commits plus the target's own next token as a bonus. Greedy-only (INFR_TEMP=0) —
    /// every committed token is checked against (or produced by) a verify-forward argmax, so
    /// the committed stream is the target's greedy stream over the VERIFY forward. That equals
    /// target-only greedy decode exactly unless a near-tie logit splits between the batched
    /// f16 verify kernels and decode's exact-f32 GEMV; end-to-end equality is pinned by
    /// `metal_spec_decode_matches_target_only_greedy`. Rollback is the session prefix-diff:
    /// rejected rows just get overwritten by the next round's suffix prefill.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_metal_spec(
        &self,
        session: &mut DenseMetalSession,
        draft: &CpuModel,
        draft_session: &mut DenseMetalSession,
        prompt: &str,
        max_new: usize,
        k: usize,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let sampler = crate::sampling::Sampler::from_env();
        if sampler.temp > 0.0 {
            return Err(anyhow!(
                "speculative decoding is greedy-only — set INFR_TEMP=0"
            ));
        }
        let vocab = self.cfg.vocab;
        let argmax = |row: &[f32]| -> u32 {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            bi as u32
        };
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let mut committed: Vec<u32> = enc.get_ids().to_vec();
        let n_prompt = committed.len();

        // Conversation slots, like the plain session chat: pick the best-prefix slot in BOTH
        // sessions from the prompt, once per call (the indices stay stable — LRU recycling only
        // happens inside pick). A returning conversation suffix-prefills its own slots; a
        // different conversation forks/seeds instead of clobbering — multi-user spec serve
        // stops paying a full re-prefill of BOTH models on every conversation switch.
        let t_slot = session.pool.pick(&session.mtl, &self.cfg, &committed)?;
        let d_slot = draft_session
            .pool
            .pick(&draft_session.mtl, &draft.cfg, &committed)?;

        // Initial fill: the target's normal (chunked-prefill) path produces the first token —
        // verify forwards are only for the small k+1 suffixes.
        let t0 = std::time::Instant::now();
        let mut acc_buf: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (first, _stats) = crate::cpu_backend::generate_dense_metal_session(
            &session.mtl,
            &self.gguf,
            &self.cfg,
            &self.token_embd,
            self.per_layer_embd.as_ref(),
            &committed,
            1,
            |_| {},
            &mut session.pool.slots[t_slot],
            session.max_ctx,
            None,
        )?;
        let prompt_secs = t0.elapsed().as_secs_f64();
        let mut t_next = *first.first().ok_or_else(|| anyhow!("empty first token"))?;
        let mut out: Vec<u32> = Vec::new();
        let ignore_eos = std::env::var("INFR_IGNORE_EOS").is_ok();
        let t1 = std::time::Instant::now();
        // Adaptive draft length: k tracks recent acceptance (EMA) — code-shaped output
        // accepts ~3/round and deserves the full k; high-entropy text accepts ~1-2 and a
        // long draft just adds verify rows that get thrown away (measured net-NEGATIVE at
        // fixed k=4 on prose). Bounds [1, k].
        let mut ema = 2.0f64;
        'outer: while out.len() < max_new {
            // Commit the token the target already chose (initial sample or verify bonus).
            let is_eos =
                !ignore_eos && (self.cfg.eos_ids.contains(&t_next) || t_next == self.cfg.eos);
            out.push(t_next);
            if !is_eos {
                stream_token(
                    &self.tokenizer,
                    &mut acc_buf,
                    &mut printed,
                    t_next,
                    &mut on_piece,
                );
            }
            committed.push(t_next);
            if is_eos || out.len() >= max_new {
                break;
            }
            // Draft proposes up to k greedy continuations of the committed stream (its session
            // suffix-prefills the divergence from last round automatically).
            let k_now = (ema.round() as usize).clamp(1, k);
            let budget = k_now.min(max_new - out.len());
            let td = std::time::Instant::now();
            let (cand, _) = crate::cpu_backend::generate_dense_metal_session(
                &draft_session.mtl,
                &draft.gguf,
                &draft.cfg,
                &draft.token_embd,
                draft.per_layer_embd.as_ref(),
                &committed,
                budget,
                |_| {},
                &mut draft_session.pool.slots[d_slot],
                draft_session.max_ctx,
                None,
            )?;
            // Verify: one batched target forward over [t_next, cand..]; row i's argmax is the
            // target's choice after consuming everything up to and including suffix row i.
            let mut feed = committed.clone();
            feed.extend_from_slice(&cand);
            if feed.len() + 1 > session.max_ctx {
                break; // context full: the committed stream is still exact
            }
            let td = td.elapsed();
            let tv = std::time::Instant::now();
            let (logits, vstats) = crate::cpu_backend::verify_dense_metal2(
                &session.mtl,
                &self.gguf,
                &self.cfg,
                &self.token_embd,
                self.per_layer_embd.as_ref(),
                &feed,
                &mut session.pool.slots[t_slot],
                session.max_ctx,
            )?;
            if std::env::var("INFR_SPEC_DEBUG").is_ok() {
                eprintln!(
                    "[spec] draft {:.1}ms verify {:.1}ms (exec {:.1}ms) cand={}",
                    td.as_secs_f64() * 1e3,
                    tv.elapsed().as_secs_f64() * 1e3,
                    vstats * 1e3,
                    cand.len()
                );
            }
            let m = logits.len() / vocab;
            // Rows cover feed's un-cached suffix; the candidate checks use the LAST cand.len()+1
            // rows (the row before cand[0] is t_next's — its argmax checks cand[0]).
            let base = m - (cand.len() + 1);
            // Target's greedy choice at each candidate row plus the bonus row, then the pure
            // accept decision (unit-tested in `spec_accept_tests`, incl. the rejection branch the
            // self-spec e2e test can't reach): the longest prefix the target ratifies and the
            // next committed token — the target's correction at the first mismatch, or the bonus
            // token when all candidates accept.
            let varg: Vec<u32> = (0..=cand.len())
                .map(|j| argmax(&logits[(base + j) * vocab..(base + j + 1) * vocab]))
                .collect();
            let (accepted, next_tok) = spec_accept(&cand, &varg);
            for &c in &cand[..accepted] {
                let is_eos = !ignore_eos && (self.cfg.eos_ids.contains(&c) || c == self.cfg.eos);
                out.push(c);
                if !is_eos {
                    stream_token(
                        &self.tokenizer,
                        &mut acc_buf,
                        &mut printed,
                        c,
                        &mut on_piece,
                    );
                }
                committed.push(c);
                if is_eos || out.len() >= max_new {
                    break 'outer;
                }
            }
            ema = 0.7 * ema + 0.3 * (accepted as f64 + 1.0);
            // Roll the committed view back past the rejected tail: the session's `cached` holds
            // all of `feed`, and the next prefix diff overwrites the stale rows.
            t_next = next_tok;
        }
        Ok(crate::GenStats {
            n_prompt,
            prompt_secs,
            n_gen: out.len(),
            decode_secs: t1.elapsed().as_secs_f64(),
        })
    }

    /// Token-level bench on the **Metal** backend through the agnostic seam (the Apple-GPU twin of
    /// [`bench`](Self::bench)): prefill `n_prompt` dummy tokens, decode `n_gen`, return the timing.
    /// Lets `infr bench` (with `INFR_METAL=1`) measure pp/tg directly comparable to `llama-bench`
    /// on the Metal build.
    /// Runs through a persistent [`DenseMetalSession`] so backend, uploaded weights, compiled
    /// pipelines, and the dequant/repack weight caches all survive across reps — a fresh backend
    /// per rep re-paid every one-time cost inside the measurement (a factored-format checkpoint
    /// re-repacked hundreds of MB per rep). The materialized tokens reset each call, so every
    /// rep still measures a FULL prefill (llama-bench keeps one context across reps the same way).
    #[cfg(target_os = "macos")]
    pub fn bench_metal(
        &self,
        session: &mut DenseMetalSession,
        n_prompt: usize,
        n_gen: usize,
    ) -> Result<crate::GenStats> {
        let prompt: Vec<u32> = (0..n_prompt.max(1)).map(|i| (i % 100) as u32).collect();
        if let Some(s) = session.pool.single().as_mut() {
            s.reset_tokens();
        }
        let (_, stats) = crate::cpu_backend::generate_dense_metal_session(
            &session.mtl,
            &self.gguf,
            &self.cfg,
            &self.token_embd,
            self.per_layer_embd.as_ref(),
            &prompt,
            n_gen,
            |_| {},
            session.pool.single(),
            session.max_ctx,
            None,
        )?;
        Ok(stats)
    }

    /// Render a user turn with the model's OWN embedded chat template (so an instruct model — Gemma,
    /// Qwen, … — answers coherently). Errors if the GGUF has no `tokenizer.chat_template` or it fails
    /// to render — infr only supports models that ship one (no fabricated-ChatML fallback).
    pub fn render_chat(&self, user: &str) -> Result<String> {
        render_chat_user(&self.gguf, &self.tokenizer, self.cfg.eos, user)
            .ok_or_else(no_template_err)
    }

    /// Render a multi-turn conversation `(role, content)` through the model's OWN embedded chat
    /// template — the [`crate::model::ChatModel::render`] primitive for the CPU dense/MoE path, so the
    /// shared [`crate::model::Chat`] can drive a history-based REPL. Same template + error contract as
    /// [`render_chat`](Self::render_chat), generalized past a single user turn.
    pub fn render_chat_messages(&self, messages: &[(&str, &str)]) -> Result<String> {
        render_chat_jinja(&self.gguf, &self.tokenizer, self.cfg.eos, messages, true)
            .ok_or_else(no_template_err)
    }

    /// Greedy generation on the CPU reference backend (no GPU). Returns the decoded text plus
    /// timing/counts ([`crate::GenStats`]) for the caller's stats line.
    /// The generated text is delivered through `on_piece` as it streams; only timing/counts are
    /// returned.
    pub fn generate_cpu(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        // Stream each generated token: incrementally detokenize and emit the new suffix.
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (_generated, stats) = crate::cpu_backend::generate_dense_cpu(
            &self.gguf,
            &self.cfg,
            &self.token_embd,
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
        )?;
        Ok(stats)
    }
}

/// A persistent CPU-reference session for DiffusionGemma's two-pass forward (Phase 2 — see
/// docs/DIFFUSIONGEMMA.md and [`CpuModel::diffusion_gemma_cpu_session`]). Model-independent (like
/// [`DenseVulkanSession`]/[`DenseMetalSession`]) — `prefill`/`denoise` take the `&CpuModel` per
/// call instead of borrowing it at construction, so a [`crate::model::ChatModel`] can hold both an
/// owned `CpuModel` and a persistent session side by side (Phase 3 — no self-referential borrow).
pub struct DiffusionGemmaCpuSession {
    be: CpuBackend,
    state: Option<crate::cpu_backend::SeamKv>,
    max_ctx: usize,
}

/// [`DiffusionGemmaCpuSession`]'s Vulkan twin (see [`CpuModel::diffusion_gemma_vulkan_session`]).
pub struct DiffusionGemmaVulkanSession {
    be: infr_vulkan::VulkanBackend,
    state: Option<crate::cpu_backend::SeamKv>,
    max_ctx: usize,
}

impl DiffusionGemmaCpuSession {
    /// Causal prefill of `tokens` (encoder scalars, chunked/per-token like every other dense
    /// prefill on this seam) — writes KV rows `0..tokens.len()`. Call once per block before
    /// [`denoise`](Self::denoise); a second call with a prompt that EXTENDS the previous one
    /// continues the session (ChatSession-style prefix reuse), matching every other session on
    /// this seam.
    pub fn prefill(&mut self, model: &CpuModel, tokens: &[u32]) -> Result<()> {
        crate::cpu_backend::generate_dense_backend(
            &self.be,
            &|_name, tb, dt, _n| match tb {
                crate::cpu_backend::WBytes::Mmap(tb) => Ok((self.be.map_weight(tb), dt)),
                crate::cpu_backend::WBytes::Owned(v) => {
                    let buf = self
                        .be
                        .alloc(v.len().max(1), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    self.be
                        .upload(buf.as_ref(), &v)
                        .map_err(|e| anyhow!("{e}"))?;
                    Ok((buf, dt))
                }
            },
            &model.gguf,
            &model.cfg,
            &model.token_embd,
            model.per_layer_embd.as_ref(),
            tokens,
            1, // rides the ordinary per-token causal-prefill loop (see `verify_dense_cpu`'s doc)
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
        )?;
        Ok(())
    }

    /// ONE canvas denoise forward over the session's already-prefilled prompt (see
    /// `DenoiseReq`'s doc): `canvas_tokens.len()` MUST match every call on this session (the
    /// model's `canvas_length`). Returns `[canvas_tokens.len() * vocab]` raw logits. `sc_logits`
    /// is the PREVIOUS step's raw canvas logits for self-conditioning (`None` = off, matching the
    /// reference's step-0 gate); `temp_inv` is the self-conditioning softmax temperature divisor
    /// (unused when `sc_logits` is `None`). Panics-free even before [`prefill`](Self::prefill) is
    /// ever called — errors instead (an empty prompt, P=0).
    pub fn denoise(
        &mut self,
        model: &CpuModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
    ) -> Result<Vec<f32>> {
        let mut out_logits = Vec::new();
        crate::cpu_backend::generate_dense_backend(
            &self.be,
            &|_name, tb, dt, _n| match tb {
                crate::cpu_backend::WBytes::Mmap(tb) => Ok((self.be.map_weight(tb), dt)),
                crate::cpu_backend::WBytes::Owned(v) => {
                    let buf = self
                        .be
                        .alloc(v.len().max(1), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    self.be
                        .upload(buf.as_ref(), &v)
                        .map_err(|e| anyhow!("{e}"))?;
                    Ok((buf, dt))
                }
            },
            &model.gguf,
            &model.cfg,
            &model.token_embd,
            model.per_layer_embd.as_ref(),
            &[], // denoise never touches the prompt/generation token stream — see `DenoiseReq`
            0,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            Some(crate::cpu_backend::DenoiseReq {
                canvas_tokens,
                sc_logits,
                temp_inv,
                out_logits: &mut out_logits,
            }),
        )?;
        Ok(out_logits)
    }
}

impl DiffusionGemmaVulkanSession {
    /// [`DiffusionGemmaCpuSession::prefill`]'s Vulkan twin.
    pub fn prefill(&mut self, model: &CpuModel, tokens: &[u32]) -> Result<()> {
        crate::cpu_backend::generate_dense_backend(
            &self.be,
            &|_name, tb, dt, _n| {
                let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
                let buf = self
                    .be
                    .alloc(padded.len(), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                self.be
                    .upload(buf.as_ref(), &padded)
                    .map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            },
            &model.gguf,
            &model.cfg,
            &model.token_embd,
            model.per_layer_embd.as_ref(),
            tokens,
            1,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
        )?;
        Ok(())
    }

    /// [`DiffusionGemmaCpuSession::denoise`]'s Vulkan twin.
    pub fn denoise(
        &mut self,
        model: &CpuModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
    ) -> Result<Vec<f32>> {
        let mut out_logits = Vec::new();
        crate::cpu_backend::generate_dense_backend(
            &self.be,
            &|_name, tb, dt, _n| {
                let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
                let buf = self
                    .be
                    .alloc(padded.len(), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                self.be
                    .upload(buf.as_ref(), &padded)
                    .map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            },
            &model.gguf,
            &model.cfg,
            &model.token_embd,
            model.per_layer_embd.as_ref(),
            &[],
            0,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            Some(crate::cpu_backend::DenoiseReq {
                canvas_tokens,
                sc_logits,
                temp_inv,
                out_logits: &mut out_logits,
            }),
        )?;
        Ok(out_logits)
    }
}

/// The speculative-decoding accept decision for one verify round (pure — no backend). Given the
/// draft `cand` and `verify_argmax` — the target verify-forward's greedy choice at each candidate
/// row plus the bonus row (`cand.len() + 1` entries; index `j` is the target's token after
/// consuming `cand[..j]`) — return the number of leading candidates the target ratifies (the
/// longest prefix where its argmax equals the draft) and the next committed token: the target's
/// correction at the first mismatch, or the bonus token when every candidate is accepted.
///
/// Correctness property (see `spec_accept_tests`): the committed stream `cand[..accepted] ++
/// [next]` is always exactly `verify_argmax[..=accepted]` — the target's own greedy stream — no
/// matter what the draft proposed. That is what makes speculative decoding output-identical to
/// target-only greedy; a wrong draft only shortens the accepted prefix, never commits a wrong
/// token. (`cfg(any(macos, test))`: the only caller is the macOS spec driver, but the logic is
/// backend-agnostic so its tests run everywhere.)
#[cfg(any(target_os = "macos", test))]
fn spec_accept(cand: &[u32], verify_argmax: &[u32]) -> (usize, u32) {
    debug_assert_eq!(verify_argmax.len(), cand.len() + 1);
    let accepted = cand
        .iter()
        .zip(verify_argmax)
        .take_while(|(c, v)| c == v)
        .count();
    (accepted, verify_argmax[accepted])
}

#[cfg(test)]
mod spec_accept_tests {
    use super::spec_accept;

    #[test]
    fn all_accepted_returns_bonus() {
        // Draft fully matches the target → accept all three, next = the bonus-row token.
        assert_eq!(spec_accept(&[10, 11, 12], &[10, 11, 12, 99]), (3, 99));
    }

    #[test]
    fn reject_at_zero_returns_correction() {
        // Draft wrong at position 0 → accept none, next = the target's correction.
        assert_eq!(spec_accept(&[10, 11, 12], &[7, 11, 12, 99]), (0, 7));
    }

    #[test]
    fn reject_in_middle_commits_prefix() {
        // Accept [10], reject at 1 → next = the correction 8; the rest is discarded.
        assert_eq!(spec_accept(&[10, 11, 12], &[10, 8, 12, 99]), (1, 8));
    }

    #[test]
    fn empty_draft_returns_sole_token() {
        // No candidates (adaptive k floored mid-round) → next = the sole bonus token.
        assert_eq!(spec_accept(&[], &[42]), (0, 42));
    }

    #[test]
    fn committed_stream_is_target_greedy_for_any_draft() {
        // The rejection-sampling invariant the self-spec e2e test cannot reach (its draft always
        // agrees, so acceptance is always full): whatever the draft proposes, the committed
        // tokens equal a prefix of the target's own argmax stream — never a token the target
        // wouldn't have produced. This is the branch the original review flagged as untested.
        let varg = [5u32, 6, 7, 8]; // the target's greedy choice at each of the four rows
        for draft in [
            vec![5, 6, 7], // perfect draft → accept 3
            vec![9, 6, 7], // wrong at 0    → accept 0
            vec![5, 9, 7], // wrong at 1    → accept 1
            vec![5, 6, 9], // wrong at 2    → accept 2
            vec![0, 0, 0], // all wrong     → accept 0
        ] {
            let (accepted, next) = spec_accept(&draft, &varg);
            let mut committed: Vec<u32> = draft[..accepted].to_vec();
            committed.push(next);
            assert_eq!(
                committed,
                varg[..=accepted].to_vec(),
                "draft {draft:?} committed a non-greedy stream"
            );
        }
    }
}
