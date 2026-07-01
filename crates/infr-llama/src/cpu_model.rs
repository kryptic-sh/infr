//! The GPU-free `CpuModel` for the CPU reference backend (no Vulkan/VRAM; weights streamed
//! from the GGUF mmap at forward time). Split out of `lib.rs` (no logic change).
use crate::*;
use anyhow::{anyhow, Result};
use infr_chat::{render_chat_jinja, render_chat_user};
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

    pub fn config(&self) -> &Config {
        &self.cfg
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

    /// Token-level bench on the **Vulkan** backend through the agnostic seam (the GPU twin of
    /// [`bench`](Self::bench)): prefill `n_prompt` dummy tokens, decode `n_gen`, return the timing.
    /// The decode tok/s here (per-token graph recompile) is the baseline the record-once replay
    /// optimization must close toward the production Recorder path.
    pub fn bench_vulkan(&self, n_prompt: usize, n_gen: usize) -> Result<crate::GenStats> {
        let prompt: Vec<u32> = (0..n_prompt.max(1)).map(|i| (i % 100) as u32).collect();
        let vk = infr_vulkan::VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        let (_, stats) = crate::cpu_backend::generate_dense_gpu(
            &vk,
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
