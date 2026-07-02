//! The Llama transformer: weight loading + all the hand-written GPU/CPU forward variants
//! (prefill, resident KV decode, record-once decode, MoE) and the multi-turn `ChatSession`.
//! Split out of `lib.rs` (no logic change); covers llama/qwen3/gemma3/gemma4/qwen3moe.
#![allow(clippy::needless_range_loop)]
use crate::*;
use anyhow::{anyhow, bail, Result};
use infr_chat::{render_chat_jinja, render_chat_oai, render_chat_user, ChatMessage};
use infr_core::backend::{Buffer, BufferUsage};
use infr_core::{Backend, WeightSource};
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;
use std::path::Path;
use tokenizers::Tokenizer;

struct LayerWeights {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn_norm_buf: Box<dyn Buffer>,
    ffn_norm_buf: Box<dyn Buffer>,
    wq: Wt,
    wk: Wt,
    /// V projection. `None` on gemma4's full-attention layers, which reuse the raw K projection as V
    /// (then a weightless RMSNorm, no RoPE). Always `Some` for every other model/layer.
    wv: Option<Wt>,
    wo: Wt,
    ffn: FfnWt,
    q_norm_buf: Option<Box<dyn Buffer>>, // qwen3 QK-norm weights [head_dim]
    k_norm_buf: Option<Box<dyn Buffer>>,
    // gemma sandwich norms: an extra RMSNorm on the attention / FFN sublayer output BEFORE the
    // residual add (`post_attention_norm` / `post_ffw_norm`). `None` for llama/qwen3.
    post_attn_norm_buf: Option<Box<dyn Buffer>>,
    post_ffw_norm_buf: Option<Box<dyn Buffer>>,
    /// gemma4 per-layer output scale (`layer_output_scale.weight`, a single scalar ~0.005–0.9): the
    /// whole layer output is multiplied by this before the next layer. `None` for other models.
    out_scale: Option<f32>,
    /// gemma4 E2B per-layer input-embedding weights (gemma3n mechanism). `None` unless the model has
    /// per-layer embeddings. `inp_gate` [n_embd→npl] gates the layer output, multiplied by the layer's
    /// per-layer input slice (GeGLU), `proj` [npl→n_embd] projects back, `post_norm` RMSNorms it.
    pl_inp_gate: Option<Wt>,
    pl_proj: Option<Wt>,
    pl_post_norm: Option<Box<dyn Buffer>>,
}

impl LayerWeights {
    /// The V-projection weight, panicking if absent — valid for every layer except gemma4's full
    /// layers (the gemma4 path checks `self.wv.is_none()` and reuses K instead).
    fn wv(&self) -> &Wt {
        self.wv
            .as_ref()
            .expect("layer has no V projection (gemma4 full layer)")
    }
    fn wgateup(&self) -> &Wt {
        match &self.ffn {
            FfnWt::Dense { wgateup, .. } => wgateup,
            FfnWt::Moe { .. } => panic!("MoE layer has no dense wgateup"),
        }
    }
    fn wdown(&self) -> &Wt {
        match &self.ffn {
            FfnWt::Dense { wdown, .. } => wdown,
            FfnWt::Moe { .. } => panic!("MoE layer has no dense wdown"),
        }
    }
    fn moe(&self) -> (&Wt, &[ExpertWt]) {
        match &self.ffn {
            FfnWt::Moe {
                gate_inp, experts, ..
            } => (gate_inp, experts),
            FfnWt::Dense { .. } => panic!("dense layer has no MoE bank"),
        }
    }
    /// The router weight + stacked expert bank, when this layer is a fully-GPU native MoE layer
    /// (the GPU-resident decode/prefill path). `None` for offloaded / per-expert layers.
    fn moe_stacked(&self) -> Option<(&Wt, &MoeStacked)> {
        match &self.ffn {
            FfnWt::Moe {
                gate_inp,
                stacked: Some(s),
                ..
            } => Some((gate_inp, s)),
            _ => None,
        }
    }
}

/// gemma4 E2B (gemma3n) per-layer input-embedding global tensors. The per-(token,layer) input vector
/// is `((model_proj·scaled_embd)·1/√n_embd, RMSNorm'd) + (tok_embd_row × √npl)) × 1/√2`.
pub(crate) struct PerLayerEmbd {
    pub(crate) npl: usize,                       // per-layer embedding width (256)
    pub(crate) n_layer: usize,                   // number of layers (35)
    pub(crate) n_embd: usize,                    // model width (1536)
    pub(crate) model_proj: Vec<f32>, // [npl*n_layer rows, n_embd] host f32 (row k = the n_embd vector to dot)
    pub(crate) proj_norm: Vec<f32>,  // [npl] RMSNorm weight over the per-layer dim
    pub(crate) tok_embd_dtype: infr_core::DType, // per_layer_token_embd dtype (gathered per token from the gguf)
    pub(crate) tok_embd_row_bytes: usize,        // bytes per token row (npl*n_layer elements)
}

pub struct Llama {
    be: VulkanBackend,
    cfg: Config,
    token_embd: Vec<f32>, // [vocab, n_embd] host, for embedding gather
    lm_head: Wt,          // [vocab, n_embd] on GPU (tied to token_embd unless output.weight)
    output_norm: Vec<f32>,
    output_norm_buf: Box<dyn Buffer>,
    /// gemma4 proportional-rope frequency divisors (`rope_freqs.weight`, `[rope_dim/2]`), used by the
    /// full-attention layers only. `None` for models without proportional rope.
    rope_freqs: Option<Box<dyn Buffer>>,
    /// gemma4 E2B per-layer input-embedding global tensors. `None` unless the model has per-layer
    /// embeddings (only E2B/E4B gemma4 variants).
    per_layer_embd: Option<PerLayerEmbd>,
    layers: Vec<LayerWeights>,
    pub(crate) tokenizer: Tokenizer,
    sampler: std::cell::Cell<Sampler>,
    /// MoE: `INFR_MOE_STREAM` makes host-offloaded (`INFR_NCMOE`) layers stream their active experts
    /// into a VRAM pool + GPU-compute instead of CPU matvec.
    moe_stream: bool,
    /// The model's GGUF, kept mmap-alive so host-backed MoE experts can read their bytes on demand
    /// (zero-copy from the OS page cache) instead of duplicating them into RAM.
    gguf: Gguf,
}

impl Llama {
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Load with an explicit HF `tokenizer.json` sidecar.
    pub fn load(gguf_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        Self::load_opt(gguf_path, Some(tokenizer_path))
    }

    /// Load deriving the tokenizer from the GGUF's embedded vocab (`tokenizer.ggml.*`) — no
    /// sidecar needed (e.g. for `ollama:` refs, whose content-addressed blobs have no
    /// `tokenizer.json` beside them).
    pub fn load_embedded(gguf_path: &Path) -> Result<Self> {
        Self::load_opt(gguf_path, None)
    }

    /// Load with an optional sidecar tokenizer; falls back to the GGUF's embedded vocab.
    pub fn load_opt(gguf_path: &Path, tokenizer_path: Option<&Path>) -> Result<Self> {
        let g = Gguf::open(gguf_path).map_err(|e| anyhow!("open gguf: {e}"))?;
        // Config is parsed purely from metadata/tensor headers (no GPU). The locals below are the
        // subset the GPU weight-loading path references; `cfg` itself is moved into the model.
        let mut cfg = Config::from_gguf(&g)?;
        let n_layer = cfg.n_layer;
        let n_embd = cfg.n_embd;
        let n_kv = cfg.n_kv;
        let head_dim = cfg.head_dim;
        let n_embd_per_layer = cfg.n_embd_per_layer;
        let qk_norm = cfg.qk_norm;
        let gemma = cfg.gemma;
        let moe = cfg.moe;
        // INFR_NCMOE=N: keep the first N layers' experts in host RAM (saves their VRAM, cf. llama.cpp
        // --n-cpu-moe). An explicit value disables the VRAM auto-fit below.
        let ncmoe_explicit = std::env::var("INFR_NCMOE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let mut n_cpu_moe = ncmoe_explicit.unwrap_or(0).min(n_layer);
        let mut moe_stream = std::env::var("INFR_MOE_STREAM").is_ok();

        let be = VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;

        // Pre-flight VRAM check: size the resident weights up front and verify they fit before
        // uploading any tensor — turns a cryptic mid-load allocator OOM into a clear early error.
        // (KV cache + activation scratch are allocated later by `new_kv`/the forward, not here.)
        let fp = weight_footprint(&g);
        let vram = be.vram();
        let gb = |b: u64| b as f64 / 1e9;
        // GPU KV cache footprint at the target context: f16 K+V per layer. Defaults to the model's
        // own trained context length (`<arch>.context_length`); override with INFR_MAX_CTX. MoE
        // attention now stores KV in VRAM, so it competes with experts for space.
        let target_ctx = std::env::var("INFR_MAX_CTX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(cfg.n_ctx_train);
        let kv_bytes =
            (n_kv * head_dim * 2/*K+V*/ * 2/*f16*/ * n_layer) as u64 * (target_ctx + 64) as u64;
        const ACT_HEADROOM: u64 = 512 * 1024 * 1024; // activation scratch + streaming pool slack
                                                     // MoE auto-fit (default; skipped if INFR_NCMOE is set): keep as many whole expert-layers on
                                                     // the GPU as fit alongside the dense weights, the ctx KV cache, and scratch — offload the
                                                     // overflow. Forced offload defaults to streaming (GPU-via-pool, ~10x the CPU path).
        if moe.is_some() && ncmoe_explicit.is_none() {
            let per_layer = (fp.expert / n_layer.max(1) as u64).max(1);
            let budget = vram
                .available
                .saturating_sub(fp.dense + kv_bytes + ACT_HEADROOM);
            let gpu_layers = (budget / per_layer).min(n_layer as u64) as usize;
            n_cpu_moe = n_layer - gpu_layers;
            if n_cpu_moe > 0 {
                moe_stream = true;
            }
            eprintln!(
                "MoE auto-fit: {gpu_layers}/{n_layer} expert layers on GPU, {n_cpu_moe} {} \
                 (ctx={target_ctx} → KV {:.2} GB)",
                if n_cpu_moe == 0 {
                    "all resident"
                } else if moe_stream {
                    "streamed"
                } else {
                    "on CPU"
                },
                gb(kv_bytes),
            );
        }
        // Experts of the first `n_cpu_moe` layers live in host RAM → subtract their
        // (uniform-per-layer) share from the VRAM total. The router/dense weights stay on GPU.
        let cpu_expert_bytes = if n_layer > 0 {
            fp.expert * n_cpu_moe as u64 / n_layer as u64
        } else {
            0
        };
        let gpu_total = fp.total() - cpu_expert_bytes;
        let experts = if fp.expert > 0 {
            let cpu = if n_cpu_moe > 0 {
                format!(
                    ", {n_cpu_moe} layers' experts on CPU = -{:.2} GB",
                    gb(cpu_expert_bytes)
                )
            } else {
                String::new()
            };
            format!(", experts {:.2} GB{cpu}", gb(fp.expert))
        } else {
            String::new()
        };
        // KV reservation only applies once the model has a GPU KV cache (MoE here; dense uses its own
        // path). Reserve it so the later `new_kv` allocation fits alongside the weights.
        let kv_reserve = if moe.is_some() { kv_bytes } else { 0 };
        eprintln!(
            "weights {:.2} GB on GPU (dense {:.2} GB{}) + KV {:.2} GB (ctx={target_ctx}) | \
             VRAM {:.2} GB {} / {:.2} GB total",
            gb(gpu_total),
            gb(fp.dense),
            experts,
            gb(kv_reserve),
            gb(vram.available),
            if vram.live { "free" } else { "total*" },
            gb(vram.total),
        );
        if gpu_total + kv_reserve + ACT_HEADROOM > vram.available {
            bail!(
                "weights {:.2} GB + KV {:.2} GB + {:.0} MB scratch exceed the {:.2} GB VRAM available \
                 (total {:.2} GB) — use a smaller quant/ctx, free GPU memory, or set INFR_NCMOE",
                gb(gpu_total),
                gb(kv_reserve),
                ACT_HEADROOM as f64 / 1e6,
                gb(vram.available),
                gb(vram.total),
            );
        }
        // Reserve the GPU-resident weight VRAM up front as one contiguous bump arena (frees in one
        // shot, no per-tensor fragmentation). Best-effort: if the contiguous block can't be obtained,
        // fall back to per-tensor allocation rather than failing the load.
        if let Err(e) = be.reserve_weights(gpu_total) {
            eprintln!("note: weight arena reservation failed ({e}); using per-tensor allocation");
        }

        // token embeddings (host) + lm head (GPU). tied unless output.weight present.
        let (token_embd, _te_shape) = load_tensor_dequant(&g, "token_embd.weight")?;
        let lm_head = if g.tensors().iter().any(|t| t.name == "output.weight") {
            upload_wt(&be, &g, "output.weight")?
        } else {
            // tied: the lm head IS token_embd — upload it raw (native blocks for quant, f16 for
            // float) for an in-shader-dequant projection. No host dequant→f16 copy, and the GPU
            // tensor stays at the native bit-width (a big VRAM win for large-vocab heads). The host
            // keeps its own f32 `token_embd` for the input-embedding gather.
            upload_wt(&be, &g, "token_embd.weight")?
        };

        let (output_norm, _) = load_tensor_dequant(&g, "output_norm.weight")?;
        let output_norm_buf = be
            .upload_weight(&output_norm)
            .map_err(|e| anyhow!("upload output_norm: {e}"))?;
        // gemma4 proportional-rope freq divisors (used by full-attention layers); absent otherwise.
        let rope_freqs = if g.tensors().iter().any(|t| t.name == "rope_freqs.weight") {
            Some(
                be.upload_weight(&load_tensor_dequant(&g, "rope_freqs.weight")?.0)
                    .map_err(|e| anyhow!("upload rope_freqs: {e}"))?,
            )
        } else {
            None
        };
        // gemma4 E2B global per-layer-embd tensors (host f32, no GPU — shared with the CPU loader).
        let per_layer_embd = build_per_layer_embd(&g, &cfg)?;

        // Loading the per-layer weights (dequant + GPU upload) dominates startup, especially for
        // big models — show a byte-progress bar so it reports copy speed + ETA (same shared style as
        // the download bar). Total/inc are GGUF source bytes; per-layer = sum of that layer's tensors.
        // Weight-load progress: every `BufferUsage::Weights` alloc advances it automatically (the
        // ticking lives in `VulkanBackend::alloc`), so the loader just opens the scope. `gpu_total`
        // is the resident VRAM denominator (CPU/host experts excluded). Drops at end of `load`.
        let _wp = be.weight_progress(Some(gpu_total));
        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let up = |be: &VulkanBackend, name: String| -> Result<Wt> { upload_wt(be, &g, &name) };
            let attn_norm = load_tensor_dequant(&g, &p("attn_norm.weight"))?.0;
            let ffn_norm = load_tensor_dequant(&g, &p("ffn_norm.weight"))?.0;
            let attn_norm_buf = be
                .upload_weight(&attn_norm)
                .map_err(|e| anyhow!("upload attn_norm {l}: {e}"))?;
            let ffn_norm_buf = be
                .upload_weight(&ffn_norm)
                .map_err(|e| anyhow!("upload ffn_norm {l}: {e}"))?;
            // MoE layer: router + per-expert bank. Dense layer: fused gate‖up + down.
            let ffn = if let Some(mc) = moe {
                load_moe(
                    &be,
                    &g,
                    &format!("blk.{l}."),
                    mc.n_expert,
                    l < n_cpu_moe,
                    n_cpu_moe == 0,
                    mc.n_ff_exp * n_embd,
                )?
            } else {
                FfnWt::Dense {
                    wgateup: build_wgateup(&be, &g, &format!("blk.{l}."))?,
                    wdown: up(&be, p("ffn_down.weight"))?,
                }
            };
            let (q_norm_buf, k_norm_buf) = if qk_norm {
                (
                    Some(
                        be.upload_weight(&load_tensor_dequant(&g, &p("attn_q_norm.weight"))?.0)
                            .map_err(|e| anyhow!("upload q_norm {l}: {e}"))?,
                    ),
                    Some(
                        be.upload_weight(&load_tensor_dequant(&g, &p("attn_k_norm.weight"))?.0)
                            .map_err(|e| anyhow!("upload k_norm {l}: {e}"))?,
                    ),
                )
            } else {
                (None, None)
            };
            // gemma sandwich norms: post-attention + post-ffw RMSNorm weights.
            let (post_attn_norm_buf, post_ffw_norm_buf) = if gemma {
                (
                    Some(
                        be.upload_weight(
                            &load_tensor_dequant(&g, &p("post_attention_norm.weight"))?.0,
                        )
                        .map_err(|e| anyhow!("upload post_attention_norm {l}: {e}"))?,
                    ),
                    Some(
                        be.upload_weight(&load_tensor_dequant(&g, &p("post_ffw_norm.weight"))?.0)
                            .map_err(|e| anyhow!("upload post_ffw_norm {l}: {e}"))?,
                    ),
                )
            } else {
                (None, None)
            };
            // gemma4 per-layer output scale: a single scalar that multiplies the layer output.
            let out_scale = if g
                .tensors()
                .iter()
                .any(|t| t.name == p("layer_output_scale.weight"))
            {
                load_tensor_dequant(&g, &p("layer_output_scale.weight"))?
                    .0
                    .first()
                    .copied()
            } else {
                None
            };
            // gemma4 E2B per-layer input-embedding weights (gate / proj / post-norm). All present iff
            // the model has per-layer embeddings; absent on dense gemma4-12b.
            let (pl_inp_gate, pl_proj, pl_post_norm) = if n_embd_per_layer > 0 {
                (
                    Some(up(&be, p("inp_gate.weight"))?),
                    Some(up(&be, p("proj.weight"))?),
                    Some(
                        be.upload_weight(&load_tensor_dequant(&g, &p("post_norm.weight"))?.0)
                            .map_err(|e| anyhow!("upload post_norm {l}: {e}"))?,
                    ),
                )
            } else {
                (None, None, None)
            };
            layers.push(LayerWeights {
                attn_norm,
                ffn_norm,
                attn_norm_buf,
                ffn_norm_buf,
                wq: up(&be, p("attn_q.weight"))?,
                wk: up(&be, p("attn_k.weight"))?,
                // gemma4 full-attention layers omit the V projection (V = raw K). Optional load.
                wv: if g.tensors().iter().any(|t| t.name == p("attn_v.weight")) {
                    Some(up(&be, p("attn_v.weight"))?)
                } else {
                    None
                },
                wo: up(&be, p("attn_output.weight"))?,
                ffn,
                q_norm_buf,
                k_norm_buf,
                post_attn_norm_buf,
                post_ffw_norm_buf,
                out_scale,
                pl_inp_gate,
                pl_proj,
                pl_post_norm,
            });
        }

        let tokenizer = match tokenizer_path {
            Some(p) => Tokenizer::from_file(p).map_err(|e| anyhow!("load tokenizer: {e}"))?,
            None => build_tokenizer(&g)?,
        };
        // Stop on the GGUF eos plus any chat-end markers in the vocab — a chat model can emit
        // <|endoftext|> mid-turn, and stopping only on <|im_end|> lets it ramble past the answer.
        add_chat_eos(&mut cfg, &tokenizer);

        let llama = Self {
            be,
            cfg,
            token_embd,
            lm_head,
            output_norm,
            output_norm_buf,
            rope_freqs,
            per_layer_embd,
            layers,
            tokenizer,
            sampler: std::cell::Cell::new(Sampler::default()),
            moe_stream,
            gguf: g,
        };
        // Compile all GPU pipelines / first-touch state up front so any later timing (run / bench /
        // serve) measures compute, not one-time setup. Failures here would also fail real inference.
        llama.warmup()?;
        Ok(llama)
    }

    /// Run a tiny prefill + decode (+ both sampler paths) through the real forward to compile every
    /// VkPipeline and first-touch GPU state. The first use of each compute kernel lazily builds its
    /// pipeline (seconds across the whole MoE kernel set); doing it here keeps it out of timed paths.
    pub fn warmup(&self) -> Result<()> {
        // Suppress per-op profiling (INFR_PROF2) during warmup: recorders read the env at
        // construction, so without this the warmup forwards' submits pollute a subsequent bench's
        // [prof2] aggregate with prefill labels (the stage profiler does the same dance).
        let prof2 = std::env::var_os("INFR_PROF2");
        if prof2.is_some() {
            std::env::remove_var("INFR_PROF2");
        }
        let r = self.warmup_inner();
        if let Some(v) = prof2 {
            std::env::set_var("INFR_PROF2", v);
        }
        r
    }

    fn warmup_inner(&self) -> Result<()> {
        let prompt: Vec<u32> = (0..64).map(|i| (i % 64) as u32).collect();
        if self.cfg.moe.is_some() {
            let mut kv = self.new_moe_kv(96)?;
            self.forward_moe_chunk(&[1u32], &mut kv)?; // shallow decode → non-split attention
            self.forward_moe_chunk(&prompt, &mut kv)?; // prefill: flash attn, routing, gather/scatter, mmq/gemv, accumulate
            self.forward_moe_chunk(&[1u32], &mut kv)?; // deep decode → split-K attn, multi-slot FFN, top-k
            let greedy = SampleParams {
                temp: 0.0,
                top_k: 1,
                top_p: 1.0,
                u: 0.0,
            };
            self.forward_moe_chunk_g(&[1u32], &mut kv, Some(greedy))?; // argmax
            let stoch = SampleParams {
                temp: 0.6,
                top_k: 20,
                top_p: 0.95,
                u: 0.5,
            };
            self.forward_moe_chunk_g(&[1u32], &mut kv, Some(stoch))?; // moe_sample (radix top-k)
        } else {
            let mut kv = self.new_kv(96)?;
            self.forward_resident_kv(&[1u32], &mut kv)?;
            self.forward_resident_kv(&prompt, &mut kv)?;
            self.forward_resident_kv(&[1u32], &mut kv)?;
        }
        Ok(())
    }

    /// Render a conversation with the model's OWN embedded chat template (`tokenizer.chat_template`,
    /// a Jinja2 string) via minijinja — the source of truth for turn markers, system handling, etc.
    /// Returns `None` if the GGUF has no template or it fails to render (the caller errors — there is
    /// no hardcoded fallback). `messages` are `(role, content)`; `bos_token`/`eos_token`
    /// come from the GGUF special-token ids.
    pub(crate) fn render_chat_messages(
        &self,
        messages: &[(&str, &str)],
        add_generation_prompt: bool,
    ) -> Option<String> {
        render_chat_jinja(
            &self.gguf,
            &self.tokenizer,
            self.cfg.eos,
            messages,
            add_generation_prompt,
        )
    }

    /// Encode a chat-template-rendered string to token ids (special markers parsed atomically; no
    /// extra auto-BOS — the template already emits `<bos>`).
    fn encode_special(&self, s: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(s, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| anyhow!("encode: {e}"))
    }

    /// Set token sampling (temp ≤ 0 → greedy). Applies to subsequent `generate`/`ChatSession::turn`.
    pub fn set_sampling(&self, temp: f32, top_k: usize, top_p: f32) {
        self.sampler.set(Sampler { temp, top_k, top_p });
    }

    fn lin(&self, w: &dyn Buffer, x: &[f32], rows: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        self.be.linear(w, x, rows, in_f, out_f).expect("gpu linear")
    }

    /// Full forward over `tokens`; returns logits (`vocab`) for the LAST position.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.cfg;
        let t = tokens.len();
        let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv, c.head_dim);

        // embedding gather -> hidden [T, n_embd]
        let mut hidden = vec![0f32; t * ne];
        for (i, &tok) in tokens.iter().enumerate() {
            let src = &self.token_embd[tok as usize * ne..(tok as usize + 1) * ne];
            hidden[i * ne..(i + 1) * ne].copy_from_slice(src);
        }

        for layer in &self.layers {
            // --- attention ---
            let hn = rmsnorm_rows(&hidden, &layer.attn_norm, t, ne, c.rms_eps);
            let mut q = self.lin(layer.wq.f16(), &hn, t, ne, nh * hd);
            let mut k = self.lin(layer.wk.f16(), &hn, t, ne, nkv * hd);
            let v = self.lin(layer.wv().f16(), &hn, t, ne, nkv * hd);
            rope_rows(&mut q, t, nh, hd, c.rope_dim, c.rope_theta);
            rope_rows(&mut k, t, nkv, hd, c.rope_dim, c.rope_theta);
            let attn = attention(&q, &k, &v, t, nh, nkv, hd);
            let ao = self.lin(layer.wo.f16(), &attn, t, nh * hd, ne);
            for i in 0..t * ne {
                hidden[i] += ao[i];
            }

            // --- ffn (SwiGLU) ---
            let hn2 = rmsnorm_rows(&hidden, &layer.ffn_norm, t, ne, c.rms_eps);
            let gu = self.lin(layer.wgateup().f16(), &hn2, t, ne, 2 * c.n_ff);
            let mut act = vec![0f32; t * c.n_ff];
            for r in 0..t {
                for i in 0..c.n_ff {
                    let g = gu[r * 2 * c.n_ff + i];
                    act[r * c.n_ff + i] = silu(g) * gu[r * 2 * c.n_ff + c.n_ff + i];
                }
            }
            let down = self.lin(layer.wdown().f16(), &act, t, c.n_ff, ne);
            for i in 0..t * ne {
                hidden[i] += down[i];
            }
        }

        // final norm on the last row, then lm_head
        let last = &hidden[(t - 1) * ne..t * ne];
        let normed = rmsnorm_rows(last, &self.output_norm, 1, ne, c.rms_eps);
        self.lin(self.lm_head.f16(), &normed, 1, ne, c.vocab)
    }

    /// Eager GPU GEMV `y = x·Wᵀ` for any weight kind (f16 / unified-Q / native), one submit. Uploads
    /// `x`, runs the matching recorder op, reads back `y`. Used by the MoE forward (many small,
    /// data-dependent matmuls that can't be baked into one resident command buffer).
    fn gemv_wt(
        &self,
        w: &Wt,
        x: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(x.len(), rows * in_f);
        let xb = self
            .be
            .alloc((rows * in_f).max(1) * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(xb.as_ref(), bytemuck::cast_slice(x))
            .map_err(|e| anyhow!("{e}"))?;
        let yb = self
            .be
            .alloc((rows * out_f).max(1) * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        match w {
            Wt::F16(b) => rec.linear(b.as_ref(), xb.as_ref(), yb.as_ref(), rows, in_f, out_f),
            Wt::Native { buf, dtype } => rec.linear_native(
                *dtype,
                buf.as_ref(),
                xb.as_ref(),
                yb.as_ref(),
                rows,
                in_f,
                out_f,
            ),
        }
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        let mut out = vec![0u8; rows * out_f * 4];
        self.be
            .download(yb.as_ref(), &mut out)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(bytemuck::cast_slice(&out).to_vec())
    }

    /// Batched eager GEMV: record many independent `y = x·Wᵀ` into ONE command buffer / submit and
    /// read them all back. Cuts per-op submit+wait latency (the MoE bottleneck — ~1400 tiny matmuls
    /// per token). Each op is `(weight, x, rows, in_f, out_f)`; returns one output vec per op.
    fn gemv_wt_many(&self, ops: &[(&Wt, &[f32], usize, usize, usize)]) -> Result<Vec<Vec<f32>>> {
        let mut xbufs = Vec::with_capacity(ops.len());
        let mut ybufs = Vec::with_capacity(ops.len());
        for &(_, x, rows, in_f, _) in ops {
            debug_assert_eq!(x.len(), rows * in_f);
            let xb = self
                .be
                .alloc((x.len()).max(1) * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            self.be
                .upload(xb.as_ref(), bytemuck::cast_slice(x))
                .map_err(|e| anyhow!("{e}"))?;
            xbufs.push(xb);
        }
        for &(_, _, rows, _, out_f) in ops {
            let yb = self
                .be
                .alloc((rows * out_f).max(1) * 4, BufferUsage::Readback)
                .map_err(|e| anyhow!("{e}"))?;
            ybufs.push(yb);
        }
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        for (i, &(w, _, rows, in_f, out_f)) in ops.iter().enumerate() {
            let (xb, yb) = (xbufs[i].as_ref(), ybufs[i].as_ref());
            match w {
                Wt::F16(b) => rec.linear(b.as_ref(), xb, yb, rows, in_f, out_f),
                Wt::Native { buf, dtype } => {
                    rec.linear_native(*dtype, buf.as_ref(), xb, yb, rows, in_f, out_f)
                }
            }
        }
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        let mut outs = Vec::with_capacity(ops.len());
        for (i, &(_, _, rows, _, out_f)) in ops.iter().enumerate() {
            let mut o = vec![0u8; rows * out_f * 4];
            self.be
                .download(ybufs[i].as_ref(), &mut o)
                .map_err(|e| anyhow!("{e}"))?;
            outs.push(bytemuck::cast_slice(&o).to_vec());
        }
        Ok(outs)
    }

    /// One-shot MoE forward over `tokens` (fresh cache) — returns last-position logits. Thin wrapper
    /// over [`forward_moe_chunk`](Self::forward_moe_chunk); used for tests / single-logit checks.
    pub fn forward_moe(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut kv = self.new_moe_kv(tokens.len() + 8)?;
        self.forward_moe_chunk(tokens, &mut kv)
    }

    /// GPU-resident forward: records the whole stack into one command buffer (one submit),
    /// all ops on the GPU. Returns logits (`vocab`) for the last position. Much fewer
    /// GPU round-trips than `forward` (which submits per linear).
    pub fn forward_resident(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let t = tokens.len();
        let (ne, nh, nkv, hd, nff) = (c.n_embd, c.n_head, c.n_kv, c.head_dim, c.n_ff);

        let mut hidden_host = vec![0f32; t * ne];
        for (i, &tok) in tokens.iter().enumerate() {
            hidden_host[i * ne..(i + 1) * ne]
                .copy_from_slice(&self.token_embd[tok as usize * ne..(tok as usize + 1) * ne]);
        }

        let alloc = |n: usize, usage: BufferUsage| -> Result<Box<dyn Buffer>> {
            self.be
                .alloc((n * 4).max(4), usage)
                .map_err(|e| anyhow!("{e}"))
        };
        let hidden = alloc(t * ne, BufferUsage::Staging)?;
        self.be
            .upload(hidden.as_ref(), bytemuck::cast_slice(&hidden_host))
            .map_err(|e| anyhow!("{e}"))?;
        let hn = alloc(t * ne, BufferUsage::Activations)?;
        let q = alloc(t * nh * hd, BufferUsage::Activations)?;
        let k = alloc(t * nkv * hd, BufferUsage::Activations)?;
        let v = alloc(t * nkv * hd, BufferUsage::Activations)?;
        let attn = alloc(t * nh * hd, BufferUsage::Activations)?;
        let ao = alloc(t * ne, BufferUsage::Activations)?;
        let hn2 = alloc(t * ne, BufferUsage::Activations)?;
        let gu = alloc(t * 2 * nff, BufferUsage::Activations)?;
        let act = alloc(t * nff, BufferUsage::Activations)?;
        let down = alloc(t * ne, BufferUsage::Activations)?;
        let logits = alloc(t * c.vocab, BufferUsage::Readback)?;

        let prof = std::env::var("INFR_PROFILE").is_ok();
        let t_rec0 = std::time::Instant::now();
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        for layer in &self.layers {
            rec.rmsnorm(
                hidden.as_ref(),
                layer.attn_norm_buf.as_ref(),
                hn.as_ref(),
                t,
                ne,
                c.rms_eps,
            );
            rec.linear(layer.wq.f16(), hn.as_ref(), q.as_ref(), t, ne, nh * hd);
            rec.linear(layer.wk.f16(), hn.as_ref(), k.as_ref(), t, ne, nkv * hd);
            rec.linear(layer.wv().f16(), hn.as_ref(), v.as_ref(), t, ne, nkv * hd);
            rec.rope(
                q.as_ref(),
                q.as_ref(),
                t,
                nh,
                hd,
                c.rope_dim,
                c.rope_theta,
                0,
            );
            rec.rope(
                k.as_ref(),
                k.as_ref(),
                t,
                nkv,
                hd,
                c.rope_dim,
                c.rope_theta,
                0,
            );
            rec.attention(
                q.as_ref(),
                k.as_ref(),
                v.as_ref(),
                attn.as_ref(),
                t,
                nh,
                nkv,
                hd,
            );
            rec.linear(layer.wo.f16(), attn.as_ref(), ao.as_ref(), t, nh * hd, ne);
            rec.add(hidden.as_ref(), ao.as_ref(), hidden.as_ref(), t * ne);
            rec.rmsnorm(
                hidden.as_ref(),
                layer.ffn_norm_buf.as_ref(),
                hn2.as_ref(),
                t,
                ne,
                c.rms_eps,
            );
            rec.linear(
                layer.wgateup().f16(),
                hn2.as_ref(),
                gu.as_ref(),
                t,
                ne,
                2 * nff,
            );
            rec.silu_mul_fused(gu.as_ref(), act.as_ref(), t, nff);
            rec.linear(layer.wdown().f16(), act.as_ref(), down.as_ref(), t, nff, ne);
            rec.add(hidden.as_ref(), down.as_ref(), hidden.as_ref(), t * ne);
        }
        rec.rmsnorm(
            hidden.as_ref(),
            self.output_norm_buf.as_ref(),
            hn.as_ref(),
            t,
            ne,
            c.rms_eps,
        );
        rec.linear(
            self.lm_head.f16(),
            hn.as_ref(),
            logits.as_ref(),
            t,
            ne,
            c.vocab,
        );
        let t_rec = t_rec0.elapsed();
        let t_gpu0 = std::time::Instant::now();
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        let t_gpu = t_gpu0.elapsed();

        let mut bytes = vec![0u8; t * c.vocab * 4];
        self.be
            .download(logits.as_ref(), &mut bytes)
            .map_err(|e| anyhow!("{e}"))?;
        if prof {
            eprintln!("[prof] t={t} record={t_rec:?} gpu_submit_wait={t_gpu:?}");
        }
        let all: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        Ok(all[(t - 1) * c.vocab..].to_vec())
    }

    /// Allocate a KV cache with room for `max_ctx` tokens.
    pub fn new_kv(&self, max_ctx: usize) -> Result<KvCache> {
        let c = &self.cfg;
        let mut k = Vec::with_capacity(c.n_layer);
        let mut v = Vec::with_capacity(c.n_layer);
        for li in 0..c.n_layer {
            // f16 KV cache: 2 bytes/elem (half the f32 footprint that grows with context). gemma4's
            // SWA and full layers have different KV row widths, so size each layer independently.
            let row = c.layer_n_kv(li) * c.layer_head_dim(li);
            k.push(
                self.be
                    .alloc((max_ctx + 64) * row * 2, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            );
            v.push(
                self.be
                    .alloc((max_ctx + 64) * row * 2, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            );
        }
        // Zero the freshly-allocated KV cache: it comes from RECYCLED VRAM, and the prefill attention
        // reads the KV padded up to a tile boundary (kv_len → next multiple of the kernel's KV tile),
        // so the padding rows beyond the real tokens are attended. On a reused buffer those rows hold a
        // PRIOR forward's stale K/V (a same-size 2nd request — e.g. the server's 2nd turn) → the
        // attention mixes in garbage → repeated-token output. Fresh/one-shot runs got zeroed memory so
        // they were fine. (This is the ROOT CAUSE of the `infr serve` garbage.)
        {
            let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
            for buf in k.iter().chain(v.iter()) {
                rec.zero(buf.as_ref(), buf.len_bytes() / 4);
            }
            rec.finish().map_err(|e| anyhow!("{e}"))?;
        }
        // Record-once decode scratch for Qwen3-style dense models (the path reuses the `_dyn` kernels).
        let dec = if c.qk_norm {
            Some(self.build_dense_decode_scratch(max_ctx)?)
        } else {
            None
        };
        Ok(KvCache {
            k,
            v,
            len: 0,
            max_ctx,
            dec,
            rec_decode: None,
        })
    }

    /// Seed `dst`'s KV cache with the first `p` tokens of `src`'s (a per-layer device buffer copy),
    /// so a NEW conversation that shares a prefix (e.g. the system prompt) with an existing slot
    /// reuses that prefix's already-computed K/V instead of re-prefilling it — a cheap bandwidth copy
    /// vs. `p` expensive forward passes. Sets `dst.len = p`; the caller sets `dst.cached` to `src`'s
    /// matching prefix. `p` must be ≤ both caches' capacity (guaranteed: same model → same max_ctx).
    pub fn kv_copy_prefix(&self, dst: &mut KvCache, src: &KvCache, p: usize) -> Result<()> {
        let c = &self.cfg;
        if p > 0 {
            let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
            for li in 0..c.n_layer {
                // f16 KV, per-layer row (gemma4's SWA/full layers differ).
                let bytes = p * c.layer_n_kv(li) * c.layer_head_dim(li) * 2;
                rec.copy(src.k[li].as_ref(), 0, dst.k[li].as_ref(), 0, bytes);
                rec.copy(src.v[li].as_ref(), 0, dst.v[li].as_ref(), 0, bytes);
            }
            rec.finish().map_err(|e| anyhow!("{e}"))?;
        }
        dst.len = p;
        // The record-once decode buffer replays this slot's PRIOR conversation's commands; drop it so
        // decode re-records against the seeded state.
        dst.rec_decode = None;
        Ok(())
    }

    /// Allocate the persistent dense-decode scratch (single token; split-K buffers sized for the
    /// worst-case chunk count).
    fn build_dense_decode_scratch(&self, max_ctx: usize) -> Result<DenseDecodeScratch> {
        let c = &self.cfg;
        let (ne, nh, nkv, hd, nff) = (c.n_embd, c.n_head, c.n_kv, c.head_dim, c.n_ff);
        let ncm = max_ctx.div_ceil(64);
        let af = |n: usize| {
            self.be
                .alloc((n * 4).max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };
        Ok(DenseDecodeScratch {
            hidden: af(ne)?,
            hn: af(ne)?,
            qr: af(nh * hd)?,
            kr: af(nkv * hd)?,
            vr: af(nkv * hd)?,
            q_f16: self
                .be
                .alloc(nh * hd * 2, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?,
            attn: af(nh * hd)?,
            gu: af(2 * nff)?,
            act: af(nff)?,
            hlast: af(ne)?,
            logits: af(c.vocab)?,
            pm: af(nh * ncm)?,
            pl: af(nh * ncm)?,
            pacc: af(nh * ncm * hd)?,
            params: self
                .be
                .alloc(8, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?,
            emb_in: self
                .be
                .alloc(ne * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?,
        })
    }

    /// KV-cached resident forward: processes only `new_tokens` (n rows), appends their K/V to
    /// the cache, and attends over the whole cache. Returns logits for the last new token.
    pub fn forward_resident_kv(&self, new_tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let n = new_tokens.len();
        let pos = kv.len;
        // Outer dims used only for sizing the shared scratch (gemma4 `head_dim`/`n_kv` are the FULL/max
        // values). Per-layer dims are re-derived inside the layer loop.
        let (ne, nh, hd, nff) = (c.n_embd, c.n_head, c.head_dim, c.n_ff);
        if pos + n > kv.max_ctx {
            bail!("KV cache overflow: {} > {}", pos + n, kv.max_ctx);
        }
        // Record-once fast path: single-token decode of a Qwen3 dense model (record once, replay).
        if n == 1 {
            if let Some(logits) = self.forward_resident_decode_ro(new_tokens[0], kv)? {
                return Ok(logits);
            }
        }

        let mut hidden_host = vec![0f32; n * ne];
        for (i, &tok) in new_tokens.iter().enumerate() {
            hidden_host[i * ne..(i + 1) * ne]
                .copy_from_slice(&self.token_embd[tok as usize * ne..(tok as usize + 1) * ne]);
        }
        // gemma scales the input embeddings by √n_embd (done in f32 before upload).
        if c.gemma {
            let s = (ne as f32).sqrt();
            for x in hidden_host.iter_mut() {
                *x *= s;
            }
        }
        let alloc = |m: usize, u: BufferUsage| -> Result<Box<dyn Buffer>> {
            self.be.alloc((m * 4).max(4), u).map_err(|e| anyhow!("{e}"))
        };
        // Prefill (many tokens) reuses each weight across all rows → a coopmat GEMM (matmul_proj)
        // beats the per-row GEMV and lets one submit cover a big chunk without tripping the GPU
        // watchdog. Decode (n==1) and Llama stay on the fused GEMV path. GEMM writes ceil(n/64)*64
        // rows (extra rows are 0), so its output buffers are M-padded to mpad.
        // gemma4 has per-layer heterogeneous head dims (256 SWA / 512 full); route it entirely
        // through the hd-general GEMV + attention_kv path (the GEMM/flash/nonfa prefill kernels
        // assume a uniform head dim). Correctness-first; prefill is slower but right.
        // gemma4 now included: the coopmat projection GEMM handles its per-layer heterogeneous dims
        // (hd 256/512, nkv 8/1, wv=None → V=K, E2B per-layer FFN/KV-sharing) via the same per-layer
        // args the GEMV path used — see the QKV/FFN blocks below. INFR_NOGEMM restores GEMV.
        let use_gemm = c.qk_norm && n >= 64 && std::env::var("INFR_NOGEMM").is_err();
        // Register-O flash (FlashAttention-2 layout, Br=128/64) is opt-in (INFR_FLASH_REG) while it's
        // A/B'd vs the BM=64 flash; it needs mpad padded to 128 (q/attn/scratch). Its smallest tile
        // (BR=64) needs 29440 B shared — skip it on devices that can't fit that (else device-lost).
        let use_flash_reg = use_gemm
            && hd == 128
            && self.be.max_shared_memory_bytes() >= 64 * infr_vulkan::FLASH_REG_SHARED_PER_ROW
            && std::env::var("INFR_FLASH_REG").is_ok();
        // gemma prefill attention: the flash warptile is hd=128-only, but the non-FA coopmat path
        // (attn_qk → softmax → attn_pv) is hd-general (256/512), so route gemma's attention through
        // it instead of the scalar `attention_kv` GEMV. gemma4 keeps GEMV *projections*
        // (use_gemm=false) but still gets coopmat *attention* here. Per-layer hd (256 SWA / 512 full)
        // flows via push constants — no compiled variants. INFR_GEMMA_NOFA restores the fallback.
        let gemma_prefill_attn = c.gemma && n >= 64 && std::env::var("INFR_GEMMA_NOFA").is_err();
        let mpad = if use_flash_reg {
            n.div_ceil(128) * 128
        } else if use_gemm || gemma_prefill_attn {
            n.div_ceil(64) * 64
        } else {
            n
        };
        // Prefill attention has TWO interchangeable algorithms — keep BOTH; which one wins is
        // HARDWARE-dependent (the card's compute:bandwidth ratio):
        //  • flash (attention_prefill_flash, split-K, 8-warp register-blocked for hd=128): never
        //    materializes the S=[m,kv] scores buffer → far less HBM. After warptile-izing its GEMMs
        //    it now also wins on this bandwidth-rich card (+8-12% across ctx, 32k 2351→2620) AND is
        //    the right choice on bandwidth-starved cards (APUs, cut-down GPUs) / very long context.
        //  • non-FA (attn_qk → softmax → attn_pv): materializes S (more HBM) but uses big-tile
        //    (BN=256) warptile GEMMs. Fallback for hd≠128 (the flash warptile is hd=128-specialized)
        //    or via INFR_NO_FLASH.
        // Both are correctness-tested (attention_prefill_{nonfa,flash}_matches_cpu) so neither rots.
        // DEFAULT = flash for hd=128. TODO: auto-select from device bandwidth/FLOP caps.
        // The flash partial's smallest tile (bm=32) needs 29056 B of shared memory; if the device's
        // maxComputeSharedMemorySize can't fit even that, fall back to the non-FA path (else the
        // over-committed pipeline faults the GPU → device-lost). RADV 64 KB / NVIDIA 48 KB both fit.
        let use_flash = use_gemm
            && hd == 128
            && self.be.max_shared_memory_bytes() >= 32 * infr_vulkan::FLASH_SHARED_PER_ROW
            && std::env::var("INFR_NO_FLASH").is_err();
        // non-FA coopmat attention: qwen3's hd≠128 / INFR_NO_FLASH fallback, OR gemma prefill
        // (hd=256/512). gemma4 has no use_gemm, so the `gemma_prefill_attn` term brings it in.
        let nonfa = (use_gemm && !use_flash && !c.gemma) || gemma_prefill_attn;
        let hidden = alloc(n * ne, BufferUsage::Staging)?;
        self.be
            .upload(hidden.as_ref(), bytemuck::cast_slice(&hidden_host))
            .map_err(|e| anyhow!("{e}"))?;
        let hn = alloc(n * ne, BufferUsage::Activations)?;
        // q is f16 (read by the f16 attention kernels), like the KV cache. q and attn are M-padded
        // to mpad rows so the coopmat prefill attention can read/write whole 64-row tiles.
        let q = self
            .be
            .alloc((mpad * nh * hd * 2).max(4), BufferUsage::Activations)
            .map_err(|e| anyhow!("{e}"))?;
        let attn = alloc(mpad * nh * hd, BufferUsage::Activations)?;
        let act = alloc(n * nff, BufferUsage::Activations)?;
        // gate+up intermediate for the un-fused decode FFN (rmsnorm → gate/up GEMV → SwiGLU). The
        // GEMM path uses its own `gu` in `gemm_bufs`; this serves the small-batch (decode) path.
        let gu_ffn = alloc(n * 2 * nff, BufferUsage::Activations)?;
        // gemma sandwich norm scratch: the un-fused (decode/small-batch) path can't fuse the residual
        // add, so it writes the attn/ffn sublayer output here, RMSNorms it, then adds to hidden.
        let gemma_sub = if c.gemma {
            Some(alloc(n * ne, BufferUsage::Activations)?)
        } else {
            None
        };
        // gemma4 E2B per-layer input embeddings (gemma3n): compute the per-(token,layer) input vector
        // on the host once per forward, laid out layer-major `[n_layer][n][npl]` so each layer's slice
        // is contiguous, and upload it. `pl_gate`/`pl_p` are the per-layer-embd application scratch.
        let (inp_per_layer, pl_gate, pl_p) = if let Some(ple) = &self.per_layer_embd {
            let (npl, nl, nem) = (ple.npl, ple.n_layer, ple.n_embd);
            let inv_sqrt_ne = 1.0 / (nem as f32).sqrt();
            let sqrt_npl = (npl as f32).sqrt();
            let inv_sqrt2 = 1.0 / 2f32.sqrt();
            let te_bytes = self
                .gguf
                .tensor_bytes("per_layer_token_embd.weight")
                .map_err(|e| anyhow!("{e}"))?;
            // per_layer_token_embd is a VOCAB table — look up each token's row by token ID (not
            // sequence position; matches llama.cpp `ggml_get_rows`). Dequant all rows once up front so
            // the parallel matmul below doesn't redo the dequant per layer.
            let pl_tok_all: Vec<Vec<f32>> = new_tokens[..n]
                .iter()
                .map(|&tok| {
                    let r0 = tok as usize * ple.tok_embd_row_bytes;
                    dequant_block(
                        ple.tok_embd_dtype,
                        &te_bytes[r0..r0 + ple.tok_embd_row_bytes],
                    )
                })
                .collect::<Result<_>>()?;
            // inp_per_layer is layer-major [nl][n][npl], so each layer's [n*npl] slice is contiguous
            // and disjoint → parallelize the big model_proj·emb matmul across layers. This is
            // ~n·nl·npl·nem MACs; single-threaded it dominated E2B prefill (~2 s at pp512). Math is
            // bit-identical to the serial version (same per-dot accumulation order).
            let mut ipl = vec![0f32; nl * n * npl];
            {
                use rayon::prelude::*;
                let (mp, pn, eps) = (&ple.model_proj, &ple.proj_norm, c.rms_eps);
                ipl.par_chunks_mut(n * npl)
                    .enumerate()
                    .for_each(|(layer, lslice)| {
                        let mut proj = vec![0f32; npl];
                        for t in 0..n {
                            let emb = &hidden_host[t * ne..t * ne + ne]; // scaled token embd (√n_embd)
                                                                         // proj = (model_proj · emb) * 1/√n_embd, then RMSNorm over npl.
                            let mut ss = 0f32;
                            for (j, pj) in proj.iter_mut().enumerate() {
                                let wrow =
                                    &mp[(layer * npl + j) * nem..(layer * npl + j) * nem + nem];
                                let acc: f32 = wrow.iter().zip(emb).map(|(a, b)| a * b).sum();
                                let v = acc * inv_sqrt_ne;
                                *pj = v;
                                ss += v * v;
                            }
                            let rms = 1.0 / (ss / npl as f32 + eps).sqrt();
                            let pl_tok = &pl_tok_all[t];
                            for j in 0..npl {
                                let normed = proj[j] * rms * pn[j];
                                let tok = pl_tok[layer * npl + j] * sqrt_npl;
                                lslice[t * npl + j] = (normed + tok) * inv_sqrt2;
                            }
                        }
                    });
            }
            let buf = alloc(nl * n * npl, BufferUsage::Staging)?;
            self.be
                .upload(buf.as_ref(), bytemuck::cast_slice(&ipl))
                .map_err(|e| anyhow!("{e}"))?;
            (
                Some(buf),
                Some(alloc(n * npl, BufferUsage::Activations)?),
                Some(alloc(n * ne, BufferUsage::Activations)?),
            )
        } else {
            (None, None, None)
        };
        // Only the LAST position's logits are needed → compute lm_head for one row. (Computing all n
        // rows at long context is a huge wasted dispatch + ~n*vocab buffer that can exceed the GPU
        // watchdog and lose the device.)
        let hlast = alloc(ne, BufferUsage::Activations)?;
        let logits = alloc(c.vocab, BufferUsage::Readback)?;
        // gemma4 V-norm: a unit-weight RMSNorm buffer (= x/rms) sized to the largest head dim. Built
        // once and reused for every layer's V normalization.
        let v_ones = if c.gemma4 {
            let ones = vec![1.0f32; c.max_head_dim()];
            let b = alloc(c.max_head_dim(), BufferUsage::Activations)?;
            self.be
                .upload(b.as_ref(), bytemuck::cast_slice(&ones))
                .map_err(|e| anyhow!("{e}"))?;
            Some(b)
        } else {
            None
        };
        // gemma4's SWA layers have a wider KV row (8*256=2048) than its full layers (1*512=512), so
        // the shared Q/K/V scratch is sized for the per-layer maxima (`nh*hd` already = the max since
        // `hd` is the full/largest head dim). For uniform models these equal `nkv*hd`.
        let kvrow_max = (c.n_kv * c.head_dim).max(c.n_kv_swa * c.head_dim_swa);
        // qwen3 (QK-norm) uses an un-fused attention input: raw Q/K/V projections then a separate
        // per-head RMSNorm+RoPE. (Llama uses the single fused attn_in instead.)
        let qkv_raw = if c.qk_norm {
            Some((
                alloc(mpad * nh * hd, BufferUsage::Activations)?,
                alloc(mpad * kvrow_max, BufferUsage::Activations)?,
                alloc(mpad * kvrow_max, BufferUsage::Activations)?,
            ))
        } else {
            None
        };
        // GEMM-prefill scratch: o-proj out (ao), gate/up out (gu), down out (down), all M-padded;
        // plus a tiny dummy buffer bound as scales/mins when the weight is f16 (unused there).
        let gemm_bufs = if use_gemm {
            Some((
                alloc(mpad * ne, BufferUsage::Activations)?,
                alloc(mpad * 2 * nff, BufferUsage::Activations)?,
                alloc(mpad * ne, BufferUsage::Activations)?,
                alloc(1, BufferUsage::Activations)?,
            ))
        } else {
            None
        };
        // Flash-decoding: for single-token decode, split each head's KV range across many
        // workgroups (partials in pm/pl/pacc), so attention isn't stuck on `nh` workgroups. The
        // chunk size is adaptive: a coarse fixed chunk leaves too few workgroups at low/mid context,
        // so size it to ~`nchunk_div` chunks/head (≈nh*nchunk_div workgroups) with a 64-key floor.
        // ~32 chunks/head saturates pass-1's KV bandwidth on the 7900 XTX (nh*32=512 workgroups ≫ 96
        // CUs) while HALVING pass-2 (attn_combine) work vs the old 64 — combine is a serial scan over
        // n_chunks, so fewer chunks is a pure win once pass-1 is bandwidth-bound (decode +3..6% at
        // d4k-16k, no shallow regression). Override with INFR_DECODE_NCHUNK. Reused across layers.
        let kv_len = pos + n;
        let nchunk_div = std::env::var("INFR_DECODE_NCHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&d| d > 0)
            .unwrap_or(32);
        let chunk = (kv_len / nchunk_div).clamp(64, 512);
        // Non-FA scores scratch: [nh, mpad, kv_pad] f16 (kv padded to 256 — the 8-warp attn_qk's BN;
        // the recorder uses the same padding).
        let nonfa_s = if nonfa && !use_flash {
            let kv_pad = kv_len.div_ceil(256) * 256;
            Some(
                self.be
                    .alloc(nh * mpad * kv_pad * 2, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        // Split-K PV partials: [max_splits, mpad, nh*hd] f32 (summed by attn_pv_reduce). Max 8 splits.
        // Flash split-K scratch: po=[≤8, mpad, nh, hd] f32 partials + pm/pl=[≤8, mpad, nh] f32.
        let flash_bufs = if use_flash || use_flash_reg {
            Some((
                alloc(8 * mpad * nh * hd, BufferUsage::Activations)?,
                alloc(8 * mpad * nh, BufferUsage::Activations)?,
                alloc(8 * mpad * nh, BufferUsage::Activations)?,
            ))
        } else {
            None
        };
        let nonfa_pv = if nonfa && !use_flash {
            Some(
                self.be
                    .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        // gemma (hd=256): the split-K decode kernel (attn_partial) tiles for hd≤128, so route
        // gemma decode through the hd-general `attention_kv` (no split; its TILE loop covers long kv).
        let use_split = n == 1 && kv_len > chunk && !c.gemma;
        let n_chunks = if use_split { kv_len.div_ceil(chunk) } else { 0 };
        let split_bufs = if use_split {
            Some((
                alloc(nh * n_chunks, BufferUsage::Activations)?,
                alloc(nh * n_chunks, BufferUsage::Activations)?,
                alloc(nh * n_chunks * hd, BufferUsage::Activations)?,
            ))
        } else {
            None
        };

        let prof = std::env::var("INFR_PROF").is_ok();
        let t_rec = std::time::Instant::now();
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        // weight-op dispatchers: pick the f16, quant, or native kernel based on how the weight is stored.
        let lin = |w: &Wt, x: &dyn Buffer, y: &dyn Buffer, rows: usize, inf: usize, outf: usize| {
            match w {
                Wt::F16(b) => rec.linear(b.as_ref(), x, y, rows, inf, outf),
                Wt::Native { buf, dtype } => {
                    rec.linear_native(*dtype, buf.as_ref(), x, y, rows, inf, outf)
                }
            }
        };
        let lin_add = |w: &Wt,
                       x: &dyn Buffer,
                       res: &dyn Buffer,
                       y: &dyn Buffer,
                       rows: usize,
                       inf: usize,
                       outf: usize| match w {
            Wt::F16(b) => rec.linear_add(b.as_ref(), x, res, y, rows, inf, outf),
            Wt::Native { buf, dtype } => {
                rec.linear_add_native(*dtype, buf.as_ref(), x, res, y, rows, inf, outf)
            }
        };
        // coopmat GEMM `c = a · Wᵀ` for prefill; binds the dummy buffer as scales/mins for f16.
        // Integer dp4a mmq path is DEFAULT for u4 projections (INFR_NOMMQ to disable). It keeps the
        // weight quantized (no per-GEMM dequant), which is the win at SMALL ubatch where the f16 path
        // falls back to the dequant-bound BN=64 gemm_proj and re-dequantizes the whole weight once
        // per BM-row-tile: mmq is +26..50% at ub≤512 and still +3..5% at ub=4096 (where the f16 warp
        // matmul is compute-bound). Adds a cheap quant_q8 activation pass amortized across projections.
        let mm = |w: &Wt, a: &dyn Buffer, cbuf: &dyn Buffer, rows: usize, k: usize, outf: usize| {
            let dummy = gemm_bufs.as_ref().unwrap().3.as_ref();
            match w {
                Wt::F16(b) => {
                    rec.matmul_proj(a, b.as_ref(), dummy, dummy, cbuf, rows, k, outf, 16, 0)
                }
                // Native-block prefill: coopmat tiled GEMM with in-shader dequant (decode-once per
                // weight element, reused across the row tile). Needs n%64, k%32 (all projections
                // satisfy); else fall back to the native GEMV.
                Wt::Native { buf, dtype } => {
                    if outf.is_multiple_of(64) && k.is_multiple_of(32) {
                        rec.matmul_native(*dtype, a, buf.as_ref(), cbuf, rows, k, outf)
                    } else {
                        rec.linear_native(*dtype, buf.as_ref(), a, cbuf, rows, k, outf)
                    }
                }
            }
        };
        for (li, layer) in self.layers.iter().enumerate() {
            // Per-layer dims (gemma4: SWA vs full differ in head_dim / KV-heads / rope dim+base;
            // uniform for every other model, so these shadow the outer values with the same numbers).
            let hd = c.layer_head_dim(li);
            let nkv = c.layer_n_kv(li);
            let kvrow = nkv * hd;
            let rope_dim = c.layer_rope_dim(li);
            let rope_theta = c.layer_rope_theta(li);
            // Per-layer FFN width (gemma4 E2B: 6144 / 12288; uniform `nff` elsewhere). The FFN scratch
            // is sized to the max `nff`; a narrower layer uses the leading prefix.
            let nff_l = c.layer_n_ff(li);
            // gemma4 E2B KV sharing: later layers don't compute their own K/V — they attend to an
            // earlier layer's cache. `own_kv` gates the K/V projection+store; `kv_src` is the cache to
            // read. Both are `li`/`true` for every layer of a non-sharing model.
            let noshare = std::env::var("INFR_E2B_NOSHARE").is_ok();
            let own_kv = c.has_own_kv(li) || noshare;
            let kv_src = if noshare { li } else { c.kv_src_layer(li) };
            // gemma4 attends with scale 1.0 (QK-norm controls the magnitude); everyone else 1/√hd.
            let attn_scale = if c.gemma4 {
                1.0
            } else {
                1.0 / (hd as f32).sqrt()
            };
            if let Some((qr, kr, vr)) = &qkv_raw {
                // qwen3: rmsnorm → Q/K/V projections → per-head QK-norm+RoPE (K/V into the cache)
                let rmsnorm_qkv = || {
                    rec.rmsnorm(
                        hidden.as_ref(),
                        layer.attn_norm_buf.as_ref(),
                        hn.as_ref(),
                        n,
                        ne,
                        c.rms_eps,
                    );
                };
                // Un-fused (rmsnorm + 3× subgroup GEMV) beats a fused attn_in: the fused kernel
                // recomputes the RMS sum-of-squares per output row (~2× compute), and the standalone
                // GEMV is the fast subgroup mul_mat_vec_q.
                if use_gemm {
                    rmsnorm_qkv();
                    mm(&layer.wq, hn.as_ref(), qr.as_ref(), n, ne, nh * hd);
                    // gemma4 E2B shared layers compute Q only (K/V come from `kv_src`'s cache); full
                    // layers have no wv (V = the raw K projection, copied before K gets QK-norm+RoPE).
                    // Mirror the GEMV branch so the coopmat projection handles gemma4's K/V structure.
                    if own_kv {
                        mm(&layer.wk, hn.as_ref(), kr.as_ref(), n, ne, kvrow);
                        match &layer.wv {
                            Some(wv) => mm(wv, hn.as_ref(), vr.as_ref(), n, ne, kvrow),
                            None => rec.copy(kr.as_ref(), 0, vr.as_ref(), 0, n * kvrow * 4),
                        }
                    }
                } else {
                    rmsnorm_qkv();
                    lin(&layer.wq, hn.as_ref(), qr.as_ref(), n, ne, nh * hd);
                    // gemma4 E2B shared layers compute Q only — K/V come from `kv_src`'s cache.
                    if own_kv {
                        lin(&layer.wk, hn.as_ref(), kr.as_ref(), n, ne, kvrow);
                        match &layer.wv {
                            Some(wv) => lin(wv, hn.as_ref(), vr.as_ref(), n, ne, kvrow),
                            // gemma4 full layers: V = the raw K projection (kr), copied before K gets
                            // QK-norm+RoPE (V instead gets a weightless RMSNorm, no rope, just below).
                            None => rec.copy(kr.as_ref(), 0, vr.as_ref(), 0, n * kvrow * 4),
                        }
                    }
                }
                // QK-norm + RoPE with the layer's rope dim and base (gemma dual-rope; uniform else).
                // gemma4 full-attention layers also apply proportional-rope freq_factors.
                let layer_ff = if c.gemma4 && !c.is_swa_layer(li) {
                    self.rope_freqs.as_deref()
                } else {
                    None
                };
                rec.qk_norm_rope(
                    qr.as_ref(),
                    layer.q_norm_buf.as_ref().unwrap().as_ref(),
                    q.as_ref(),
                    n,
                    nh,
                    hd,
                    rope_dim,
                    rope_theta,
                    pos,
                    0,
                    c.rms_eps,
                    layer_ff,
                );
                if own_kv {
                    rec.qk_norm_rope(
                        kr.as_ref(),
                        layer.k_norm_buf.as_ref().unwrap().as_ref(),
                        kv.k[li].as_ref(),
                        n,
                        nkv,
                        hd,
                        rope_dim,
                        rope_theta,
                        pos,
                        pos,
                        c.rms_eps,
                        layer_ff,
                    );
                    // gemma4 applies a weightless per-head RMSNorm to V before caching (rmsnorm with a
                    // unit weight = x/rms). Done in place on the f32 `vr` prior to the f16 cast-store.
                    if let Some(ones) = &v_ones {
                        rec.rmsnorm(
                            vr.as_ref(),
                            ones.as_ref(),
                            vr.as_ref(),
                            n * nkv,
                            hd,
                            c.rms_eps,
                        );
                    }
                    // v_raw is f32; cast into the f16 V cache at row offset `pos`.
                    rec.store_f16(vr.as_ref(), kv.v[li].as_ref(), n * kvrow, pos * kvrow);
                }
            } else {
                rec.attn_in(
                    hidden.as_ref(),
                    layer.attn_norm_buf.as_ref(),
                    layer.wq.f16(),
                    layer.wk.f16(),
                    layer.wv().f16(),
                    q.as_ref(),
                    kv.k[li].as_ref(),
                    kv.v[li].as_ref(),
                    n,
                    ne,
                    nh,
                    nkv,
                    hd,
                    c.rope_dim,
                    c.rope_theta,
                    pos,
                    c.rms_eps,
                );
            }
            if use_flash_reg {
                // prefill: FlashAttention-2 register-O (Br=128) — opt-in A/B vs the BM=64 flash.
                let (po, pm, pl) = flash_bufs.as_ref().unwrap();
                rec.attention_prefill_flash_reg(
                    q.as_ref(),
                    kv.k[li].as_ref(),
                    kv.v[li].as_ref(),
                    attn.as_ref(),
                    po.as_ref(),
                    pm.as_ref(),
                    pl.as_ref(),
                    n,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                );
            } else if use_flash {
                // prefill: fused flash attention (no materialized S buffer), split-K for occupancy.
                let (po, pm, pl) = flash_bufs.as_ref().unwrap();
                rec.attention_prefill_flash(
                    q.as_ref(),
                    kv.k[li].as_ref(),
                    kv.v[li].as_ref(),
                    attn.as_ref(),
                    po.as_ref(),
                    pm.as_ref(),
                    pl.as_ref(),
                    n,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                );
            } else if let Some(s) = &nonfa_s {
                // prefill: non-FA clean coopmat GEMMs (QK → softmax → PV). Handles hd=128 (qwen3),
                // hd=256/512 (gemma), the SWA window, and the scale override (gemma4 = 1.0). kv_src
                // == li except gemma4-E2B KV-sharing layers.
                rec.attention_prefill_nonfa(
                    q.as_ref(),
                    kv.k[kv_src].as_ref(),
                    kv.v[kv_src].as_ref(),
                    attn.as_ref(),
                    s.as_ref(),
                    nonfa_pv.as_ref().unwrap().as_ref(),
                    n,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                    if c.is_swa_layer(li) { c.swa_window } else { 0 },
                    attn_scale,
                );
            } else if let Some((pm, pl, pacc)) = &split_bufs {
                rec.attention_kv_split(
                    q.as_ref(),
                    kv.k[li].as_ref(),
                    kv.v[li].as_ref(),
                    attn.as_ref(),
                    pm.as_ref(),
                    pl.as_ref(),
                    pacc.as_ref(),
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    chunk,
                    n_chunks,
                    0.0,
                    0,
                );
            } else {
                rec.attention_kv(
                    q.as_ref(),
                    kv.k[kv_src].as_ref(),
                    kv.v[kv_src].as_ref(),
                    attn.as_ref(),
                    n,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                    if c.is_swa_layer(li) { c.swa_window } else { 0 },
                    attn_scale,
                );
            }
            if use_gemm {
                let (ao, gu, down, _) = gemm_bufs.as_ref().unwrap();
                // o-proj via GEMM then residual add (matmul_proj can't fuse the residual). gemma
                // inserts a post-attention RMSNorm on `ao` before the add (sandwich norm).
                mm(&layer.wo, attn.as_ref(), ao.as_ref(), n, nh * hd, ne);
                if let Some(pn) = &layer.post_attn_norm_buf {
                    rec.rmsnorm(ao.as_ref(), pn.as_ref(), ao.as_ref(), n, ne, c.rms_eps);
                }
                rec.add(hidden.as_ref(), ao.as_ref(), hidden.as_ref(), n * ne);
                // FFN un-fused: rmsnorm → gate/up GEMM → (Si|Ge)GLU → down GEMM → residual add. gemma
                // uses GeGLU and a post-ffw RMSNorm on `down` before the add.
                rec.rmsnorm(
                    hidden.as_ref(),
                    layer.ffn_norm_buf.as_ref(),
                    hn.as_ref(),
                    n,
                    ne,
                    c.rms_eps,
                );
                // nff_l = this layer's FFN width (gemma4 E2B has per-layer widths; == nff otherwise).
                mm(layer.wgateup(), hn.as_ref(), gu.as_ref(), n, ne, 2 * nff_l);
                if c.gemma {
                    rec.gelu_mul_fused(gu.as_ref(), act.as_ref(), n, nff_l);
                } else {
                    rec.silu_mul_fused(gu.as_ref(), act.as_ref(), n, nff_l);
                }
                mm(layer.wdown(), act.as_ref(), down.as_ref(), n, nff_l, ne);
                if let Some(pn) = &layer.post_ffw_norm_buf {
                    rec.rmsnorm(down.as_ref(), pn.as_ref(), down.as_ref(), n, ne, c.rms_eps);
                }
                rec.add(hidden.as_ref(), down.as_ref(), hidden.as_ref(), n * ne);
            } else if c.gemma {
                // gemma small-batch/decode: sandwich norms forbid the fused residual add, so o-proj
                // and down write to `gemma_sub`, get RMSNorm'd, then add to hidden. FFN = GeGLU.
                let sub = gemma_sub.as_ref().unwrap();
                lin(&layer.wo, attn.as_ref(), sub.as_ref(), n, nh * hd, ne);
                rec.rmsnorm(
                    sub.as_ref(),
                    layer.post_attn_norm_buf.as_ref().unwrap().as_ref(),
                    sub.as_ref(),
                    n,
                    ne,
                    c.rms_eps,
                );
                rec.add(hidden.as_ref(), sub.as_ref(), hidden.as_ref(), n * ne);
                rec.rmsnorm(
                    hidden.as_ref(),
                    layer.ffn_norm_buf.as_ref(),
                    hn.as_ref(),
                    n,
                    ne,
                    c.rms_eps,
                );
                lin(
                    layer.wgateup(),
                    hn.as_ref(),
                    gu_ffn.as_ref(),
                    n,
                    ne,
                    2 * nff_l,
                );
                rec.gelu_mul_fused(gu_ffn.as_ref(), act.as_ref(), n, nff_l);
                lin(layer.wdown(), act.as_ref(), sub.as_ref(), n, nff_l, ne);
                rec.rmsnorm(
                    sub.as_ref(),
                    layer.post_ffw_norm_buf.as_ref().unwrap().as_ref(),
                    sub.as_ref(),
                    n,
                    ne,
                    c.rms_eps,
                );
                rec.add(hidden.as_ref(), sub.as_ref(), hidden.as_ref(), n * ne);
            } else {
                lin_add(
                    &layer.wo,
                    attn.as_ref(),
                    hidden.as_ref(),
                    hidden.as_ref(),
                    n,
                    nh * hd,
                    ne,
                );
                // Un-fused FFN: rmsnorm → gate/up subgroup GEMV → SwiGLU (no per-output redundant
                // RMS sum-of-squares; reuses the fast mul_mat_vec_q).
                rec.rmsnorm(
                    hidden.as_ref(),
                    layer.ffn_norm_buf.as_ref(),
                    hn.as_ref(),
                    n,
                    ne,
                    c.rms_eps,
                );
                lin(
                    layer.wgateup(),
                    hn.as_ref(),
                    gu_ffn.as_ref(),
                    n,
                    ne,
                    2 * nff,
                );
                rec.silu_mul_fused(gu_ffn.as_ref(), act.as_ref(), n, nff);
                lin_add(
                    layer.wdown(),
                    act.as_ref(),
                    hidden.as_ref(),
                    hidden.as_ref(),
                    n,
                    nff,
                    ne,
                );
            }
            // gemma4 E2B per-layer input embeddings (gemma3n): mix this layer's per-layer input vector
            // into `hidden` AFTER the FFN residual, BEFORE the out_scale. `g = gelu(inp_gate·hidden) *
            // inp_per_layer[il]`, `p = post_norm(proj·g)`, `hidden += p` (residual on the pre-embd value).
            if let (Some(ipl), Some(gate_w), Some(proj_w), Some(post_norm)) = (
                &inp_per_layer,
                &layer.pl_inp_gate,
                &layer.pl_proj,
                &layer.pl_post_norm,
            ) {
                if std::env::var("INFR_E2B_NOPLE").is_err() {
                    let npl = self.per_layer_embd.as_ref().unwrap().npl;
                    let g = pl_gate.as_ref().unwrap();
                    let p = pl_p.as_ref().unwrap();
                    // g = inp_gate · hidden  [n_embd → npl]
                    lin(gate_w, hidden.as_ref(), g.as_ref(), n, ne, npl);
                    // g = gelu(g) * inp_per_layer[il]  (layer il's contiguous [n, npl] slice)
                    let off = li * n * npl * 4;
                    rec.gelu_mul_off(g.as_ref(), ipl.as_ref(), off, g.as_ref(), n * npl);
                    // p = proj · g  [npl → n_embd], then weightless... no: RMSNorm with post_norm.
                    lin(proj_w, g.as_ref(), p.as_ref(), n, npl, ne);
                    rec.rmsnorm(p.as_ref(), post_norm.as_ref(), p.as_ref(), n, ne, c.rms_eps);
                    // residual: hidden = hidden + p
                    rec.add(hidden.as_ref(), p.as_ref(), hidden.as_ref(), n * ne);
                }
            }
            // gemma4: multiply the whole layer output by the per-layer scalar before the next layer.
            if let Some(s) = layer.out_scale {
                rec.scale(hidden.as_ref(), s, n * ne);
            }
        }
        // final norm + lm_head on the LAST row only: copy hidden[n-1] → hlast, norm it, project.
        rec.copy(hidden.as_ref(), (n - 1) * ne * 4, hlast.as_ref(), 0, ne * 4);
        rec.rmsnorm(
            hlast.as_ref(),
            self.output_norm_buf.as_ref(),
            hn.as_ref(),
            1,
            ne,
            c.rms_eps,
        );
        lin(&self.lm_head, hn.as_ref(), logits.as_ref(), 1, ne, c.vocab);
        let rec_us = t_rec.elapsed().as_micros();
        let t_gpu = std::time::Instant::now();
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        if prof {
            eprintln!(
                "[prof] n={n} record={rec_us}us submit+wait={}us",
                t_gpu.elapsed().as_micros()
            );
        }

        let mut bytes = vec![0u8; c.vocab * 4];
        self.be
            .download(logits.as_ref(), &mut bytes)
            .map_err(|e| anyhow!("{e}"))?;
        kv.len += n;
        let mut out: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        // gemma4 final logit softcap: `logits = cap * tanh(logits / cap)`. Cheap host-side pass over
        // the single returned row (no shader needed).
        if c.final_softcap > 0.0 {
            let cap = c.final_softcap;
            for x in out.iter_mut() {
                *x = cap * (*x / cap).tanh();
            }
        }
        Ok(out)
    }

    /// Record-once single-token decode for a dense Qwen3 model — mirrors `forward_moe_chunk_gpu`: the
    /// whole forward (embed copy → 48 layers → final norm → vocab GEMV) is recorded into a replayable
    /// command buffer keyed by the attention structure; each token writes only the params SSBO + the
    /// embedding, then replays. Returns last-token vocab logits (host sampling, like the general path).
    /// Returns `None` when ineligible (non-Qwen3, no scratch, or profiling) so the caller falls back.
    fn forward_resident_decode_ro(&self, token: u32, kv: &mut KvCache) -> Result<Option<Vec<f32>>> {
        let c = &self.cfg;
        // Eligible: Qwen3 (qk-norm; per-quant QKV GEMVs) OR a Llama-arch f16 model (the fused attn_in
        // path, which requires f16 Q/K/V weights). Quantized Llama / offload / profiling fall back.
        let llama_f16 = !c.qk_norm && matches!(self.layers[0].wq, Wt::F16(_));
        // gemma (sandwich norms + GeGLU + SWA) isn't wired into the record-once decode yet — fall
        // back to forward_resident_kv, which has the full gemma path.
        if c.gemma
            || (!c.qk_norm && !llama_f16)
            || kv.dec.is_none()
            || std::env::var("INFR_PROF2").is_ok()
        {
            return Ok(None);
        }
        let (ne, nh, nkv, hd, nff) = (c.n_embd, c.n_head, c.n_kv, c.head_dim, c.n_ff);
        let kvrow = nkv * hd;
        let pos = kv.len;
        let kv_len = pos + 1;
        let dec = kv.dec.as_ref().unwrap();
        // Per-token host writes: the embedding (into the mapped emb_in the recorded buffer copies into
        // hidden) and [pos, kv_len] (the params SSBO the `_dyn` kernels read). Both are mapped, no submit.
        let emb = &self.token_embd[token as usize * ne..(token as usize + 1) * ne];
        self.be
            .upload(dec.emb_in.as_ref(), bytemuck::cast_slice(emb))
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(
                dec.params.as_ref(),
                bytemuck::cast_slice(&[pos as u32, kv_len as u32]),
            )
            .map_err(|e| anyhow!("{e}"))?;
        let (hidden, hn, qr, kr, vr, q_f16, attn, gu, act, hlast, logits, params) = (
            &dec.hidden,
            &dec.hn,
            &dec.qr,
            &dec.kr,
            &dec.vr,
            &dec.q_f16,
            &dec.attn,
            &dec.gu,
            &dec.act,
            &dec.hlast,
            &dec.logits,
            &dec.params,
        );
        let chunk = (kv_len / 32).clamp(64, 512);
        let use_split = kv_len > chunk;
        let n_chunks = if use_split { kv_len.div_ceil(chunk) } else { 0 };
        let sig = (use_split, chunk, n_chunks);
        let hit = kv.rec_decode.as_ref().is_some_and(|(s, _)| *s == sig);
        if !hit {
            let rec = self.be.recorder_persistent().map_err(|e| anyhow!("{e}"))?;
            rec.copy(dec.emb_in.as_ref(), 0, hidden.as_ref(), 0, ne * 4);
            for (li, layer) in self.layers.iter().enumerate() {
                if c.qk_norm {
                    // Qwen3: rmsnorm → per-quant Q/K/V GEMVs → QK-norm+RoPE (Q→q_f16, K→cache, V→cache).
                    rec.rmsnorm(
                        hidden.as_ref(),
                        layer.attn_norm_buf.as_ref(),
                        hn.as_ref(),
                        1,
                        ne,
                        c.rms_eps,
                    );
                    rec_linear(&rec, &layer.wq, hn.as_ref(), qr.as_ref(), 1, ne, nh * hd);
                    rec_linear(&rec, &layer.wk, hn.as_ref(), kr.as_ref(), 1, ne, kvrow);
                    rec_linear(&rec, layer.wv(), hn.as_ref(), vr.as_ref(), 1, ne, kvrow);
                    rec.qk_norm_rope_dyn(
                        qr.as_ref(),
                        layer.q_norm_buf.as_ref().unwrap().as_ref(),
                        params.as_ref(),
                        None,
                        q_f16.as_ref(),
                        1,
                        nh,
                        hd,
                        c.rope_dim,
                        c.rope_theta,
                        0,
                        c.rms_eps,
                    );
                    rec.qk_norm_rope_dyn(
                        kr.as_ref(),
                        layer.k_norm_buf.as_ref().unwrap().as_ref(),
                        params.as_ref(),
                        None,
                        kv.k[li].as_ref(),
                        1,
                        nkv,
                        hd,
                        c.rope_dim,
                        c.rope_theta,
                        1,
                        c.rms_eps,
                    );
                    rec.store_f16_dyn(vr.as_ref(), params.as_ref(), kv.v[li].as_ref(), kvrow);
                } else {
                    // Llama: one fused kernel does rmsnorm + Q/K/V proj + RoPE + KV append (f16 weights).
                    rec.attn_in_dyn(
                        hidden.as_ref(),
                        layer.attn_norm_buf.as_ref(),
                        layer.wq.f16(),
                        layer.wk.f16(),
                        layer.wv().f16(),
                        params.as_ref(),
                        q_f16.as_ref(),
                        kv.k[li].as_ref(),
                        kv.v[li].as_ref(),
                        1,
                        ne,
                        nh,
                        nkv,
                        hd,
                        c.rope_dim,
                        c.rope_theta,
                        c.rms_eps,
                    );
                }
                if use_split {
                    rec.attention_kv_split_dyn(
                        q_f16.as_ref(),
                        kv.k[li].as_ref(),
                        kv.v[li].as_ref(),
                        attn.as_ref(),
                        dec.pm.as_ref(),
                        dec.pl.as_ref(),
                        dec.pacc.as_ref(),
                        params.as_ref(),
                        nh,
                        nkv,
                        hd,
                        chunk,
                        n_chunks,
                        0.0,
                        0,
                    );
                } else {
                    rec.attention_kv_dyn(
                        q_f16.as_ref(),
                        kv.k[li].as_ref(),
                        kv.v[li].as_ref(),
                        params.as_ref(),
                        attn.as_ref(),
                        1,
                        nh,
                        nkv,
                        hd,
                        0.0,
                        0,
                    );
                }
                rec_linear_add(
                    &rec,
                    &layer.wo,
                    attn.as_ref(),
                    hidden.as_ref(),
                    hidden.as_ref(),
                    1,
                    nh * hd,
                    ne,
                );
                rec.rmsnorm(
                    hidden.as_ref(),
                    layer.ffn_norm_buf.as_ref(),
                    hn.as_ref(),
                    1,
                    ne,
                    c.rms_eps,
                );
                mixers::ffn::record_swiglu(
                    &rec,
                    hn.as_ref(),
                    mixers::ffn::GateUp::Fused(layer.wgateup()),
                    layer.wdown(),
                    gu.as_ref(),
                    gu.as_ref(), // g unused for Fused
                    gu.as_ref(), // u unused for Fused
                    act.as_ref(),
                    hidden.as_ref(),
                    Some(hidden.as_ref()), // fused residual add (in-place)
                    1,
                    ne,
                    nff,
                );
            }
            // final norm + vocab GEMV on the single row (hidden row 0 → hlast).
            rec.copy(hidden.as_ref(), 0, hlast.as_ref(), 0, ne * 4);
            rec.rmsnorm(
                hlast.as_ref(),
                self.output_norm_buf.as_ref(),
                hn.as_ref(),
                1,
                ne,
                c.rms_eps,
            );
            rec.label_next("vocab");
            rec_linear(
                &rec,
                &self.lm_head,
                hn.as_ref(),
                logits.as_ref(),
                1,
                ne,
                c.vocab,
            );
            kv.rec_decode = Some((sig, rec.finish_record().map_err(|e| anyhow!("{e}"))?));
        }
        kv.rec_decode
            .as_ref()
            .unwrap()
            .1
            .replay()
            .map_err(|e| anyhow!("{e}"))?;
        kv.len += 1;
        let mut bytes = vec![0u8; c.vocab * 4];
        self.be
            .download(dec.logits.as_ref(), &mut bytes)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(Some(bytemuck::cast_slice(&bytes).to_vec()))
    }

    /// Prefill chunk size at cache position `pos`. One chunk = one GPU submit; its cost grows with
    /// chunk×context, so a fixed chunk trips the watchdog (device-lost) at long context. Keep the
    /// per-submit work roughly constant by shrinking the chunk as context grows. Qwen3 (coopmat
    /// GEMM, cheap projections) gets a bigger budget than Llama (GEMV). Rounded to a multiple of 64
    /// for the GEMM tiling, floored at 64.
    pub fn prefill_chunk(&self, pos: usize) -> usize {
        // The coopmat prefill attention launches nh*ceil(chunk/64) workgroups; a too-small chunk
        // starves GPU occupancy (only nh=16 workgroups at chunk=64), which dominates at depth.
        // Keep chunks large — bigger chunks are more efficient PER QUERY despite re-reading KV —
        // with a min that holds occupancy up, while the budget still bounds per-submit work to stay
        // under the GPU watchdog at very long context.
        // Budget bumped 16M→32M: keeps the chunk at the 2048 cap through ~pos 15k and ~1000 at 32k
        // (was ~1000 at 16k, ~500 at 32k). Bigger chunks at depth are a free win now that prefill is
        // mmq + flash (lower per-token work) — a coding-agent turn ingests at depth, so its chunks
        // were the over-shrunk ones. 2048 chunks warmed to 32k run without tripping the watchdog on
        // this model; the budget still tapers for very long context / bigger models.
        // gemma3 + gemma4 now run prefill on coopmat: GEMM projections (matmul_native/mmq) + the
        // non-FA coopmat attention (attn_qk → softmax → attn_pv, hd-general 256/512). Both are far
        // faster than the old scalar GEMV path, but a too-large per-submit chunk still trips the GPU
        // hang-recovery watchdog (~2 s): the first overrun soft-recovers (garbage output), the second
        // consecutive one HARD-recovers → device lost. Cap the chunk so each submit stays well under
        // it; taper with context so the growing attention span stays bounded.
        if self.cfg.gemma {
            // matmul_proj (the QKVO + FFN projection GEMMs) is the DOMINANT gemma prefill op and it
            // is CHUNK-BOUND — a bigger chunk amortizes each weight read across more rows. gemma3-12b
            // pp512 matmul_proj drops 937→597 ms (426→559 t/s) going chunk 128→512. But the per-submit
            // attention span is chunk·kv ≈ chunk·(pos+chunk): letting chunk stay at 512 as `pos` grows
            // blows this past the GPU hang watchdog (a forced constant chunk=512 at pp8000 is a hard
            // device-lost). So pick the LARGEST chunk whose span stays within a proven-safe budget L:
            // chunk² + pos·chunk − L ≤ 0 → chunk ≤ (√(pos²+4L) − pos)/2. This hands out big chunks at
            // shallow depth (where the projection GEMM is most starved) and tapers exactly like the
            // old `L/(pos+1)` budget at depth. L=524 288 is BELOW the peak span the old ceil-128 path
            // already ran safely (128·4223 ≈ 540 k at pos≈4096), so it's watchdog-safe at every depth.
            // gemma4 now uses the SAME ceil as gemma3 (512): with GEMM projections it is no longer
            // GEMV-projection-bound, and its hd=512 full layers stay within the span bound (verified
            // r3 pp512/pp2048 + deep with no device-lost).
            let ceil = 512;
            let l = 524_288f64;
            let p = pos as f64;
            let chunk = 0.5 * ((p * p + 4.0 * l).sqrt() - p);
            return (chunk as usize).clamp(32, ceil);
        }
        let budget = if self.cfg.qk_norm {
            32_000_000
        } else {
            256 * 64
        };
        let raw = (budget / (pos + 1)).clamp(256, 2048);
        (raw / 64 * 64).max(64)
    }

    /// Prefill `new_tokens` into `kv` in watchdog-sized chunks, returning the last-token logits.
    fn prefill(&self, new_tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>> {
        let len = new_tokens.len();
        let mut logits = Vec::new();
        let mut i = 0;
        while i < len {
            let end = (i + self.prefill_chunk(kv.len)).min(len);
            logits = self.forward_resident_kv(&new_tokens[i..end], kv)?;
            i = end;
        }
        Ok(logits)
    }

    /// Prefill `new_tokens` into `kv`, then decode up to `max_new` tokens (stop at any EOS), streaming
    /// each decoded piece to `on_token`. Returns the generated token ids. `kv` carries the context, so
    /// repeated calls continue one conversation. The EOS token is not appended to the cache.
    fn run_in_cache(
        &self,
        new_tokens: &[u32],
        kv: &mut KvCache,
        max_new: usize,
        on_token: impl FnMut(&str),
    ) -> Result<Vec<u32>> {
        let logits = self.prefill(new_tokens, kv)?;
        self.decode_loop(logits, kv, max_new, None, on_token)
    }

    /// Greedy/sampled decode loop from `logits` (the next-token distribution), appending to `kv`.
    /// When `constraint` is set, each step masks the logits to the grammar's allowed tokens, then
    /// advances the grammar by the sampled token plus any deterministically-forced fast-forward
    /// tokens (which are appended to the output and the KV cache without being sampled). The loop
    /// ends when the grammar reaches its accepting stop state, on EOS, or at `max_new`.
    fn decode_loop(
        &self,
        mut logits: Vec<f32>,
        kv: &mut KvCache,
        max_new: usize,
        mut constraint: Option<&mut crate::grammar::Constraint>,
        mut on_token: impl FnMut(&str),
    ) -> Result<Vec<u32>> {
        let sampler = self.sampler.get();
        // xorshift64 seed (non-zero); varies per call so sampling isn't fixed across turns.
        let mut rng = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9e3779b97f4a7c15)
            | 1;
        let mut generated: Vec<u32> = Vec::new();
        // Stream UTF-8-safely: decode the whole reply each step and emit only the newly-completed
        // suffix. A multi-byte char (e.g. an emoji) is split across byte-level BPE tokens; decoding a
        // single token would yield a partial sequence → U+FFFD (the `�`). Holding until the decode no
        // longer ends in the replacement char emits whole characters only. `on_token` fires once per
        // generated token (delta may be empty while a char is mid-completion), so callers can count.
        let mut stream = StreamDecoder::default();
        let mut budget = max_new;
        while budget > 0 {
            // The tokens emitted THIS step + whether the grammar finished.
            let (step, done): (Vec<u32>, bool) = if let Some(c) = constraint.as_deref_mut() {
                // Shared llguidance step (grammar::constrained_step): drain forced tokens, else
                // mask + validate-before-commit argmax. Empty step ⇒ the constrained span ended.
                let (step, done) =
                    crate::grammar::constrained_step(c, &mut logits, &self.cfg.eos_ids)?;
                if step.is_empty() {
                    break;
                }
                (step, done)
            } else {
                let next = sample_logits(&logits, sampler, &mut rng);
                if self.cfg.eos_ids.contains(&next) {
                    break;
                }
                (vec![next], false)
            };
            generated.extend_from_slice(&step);
            let full = self.tokenizer.decode(&generated, true).unwrap_or_default();
            on_token(&stream.step(&full));
            if done {
                break;
            }
            budget = budget.saturating_sub(step.len());
            logits = self.forward_resident_kv(&step, kv)?;
        }
        Ok(generated)
    }

    /// Render a user turn with the model's OWN embedded chat template (so an instruct model answers
    /// coherently). Errors if the GGUF has no `tokenizer.chat_template` or it fails to render. Mirrors
    /// [`CpuModel::render_chat`] so the GPU and CPU golden tests feed identical token streams.
    pub fn render_chat(&self, user: &str) -> Result<String> {
        render_chat_user(&self.gguf, &self.tokenizer, self.cfg.eos, user)
            .ok_or_else(no_template_err)
    }

    /// Render a full OpenAI-shaped conversation (multi-turn, with tool calls + results) plus an
    /// optional `tools` spec through the model's embedded chat template — the tool-calling prompt
    /// path for the server. Errors if the GGUF has no chat template or it fails to render.
    pub fn render_chat_oai(
        &self,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        render_chat_oai(
            &self.gguf,
            &self.tokenizer,
            self.cfg.eos,
            messages,
            tools,
            true,
        )
        .ok_or_else(no_template_err)
    }

    /// Generate raw token ids (not a decoded string). For callers that must inspect special markers
    /// (`<think>`, `<tool_call>`) which `decode(.., skip_special=true)` would strip — e.g. the
    /// tool-calling server path. An optional grammar `constraint` masks decoding to valid tokens
    /// (reliable tool calls). Streams decoded pieces via `on_token` like [`generate`](Self::generate).
    pub fn generate_ids(
        &self,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        on_token: impl FnMut(&str),
    ) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let mut kv = self.new_kv(prompt_tokens.len() + max_new + 8)?;
        let logits = self.prefill(&prompt_tokens, &mut kv)?;
        self.decode_loop(logits, &mut kv, max_new, constraint, on_token)
    }

    /// Like [`generate_ids`] but REUSES a persistent KV cache across calls (for `infr serve`): only
    /// the token suffix that differs from `cached` (what's already materialized in `kv`) is
    /// prefilled, so a repeated prompt prefix — a coding agent's stable system prompt + growing
    /// history — is NOT re-prefilled every request. This collapses time-to-first-token from
    /// "prefill the whole prompt" to "prefill just the new turn". `kv` is lazily created sized to
    /// `INFR_MAX_CTX` / the model's trained context. Returns the generated token ids. On overflow
    /// or `INFR_NO_KV_REUSE`, prefills fresh; on any error the cache is invalidated so the next call
    /// starts clean (the cache must exactly mirror the KV, so never leave it inconsistent).
    pub fn generate_ids_cached(
        &self,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        on_token: impl FnMut(&str),
        kv: &mut Option<KvCache>,
        cached: &mut Vec<u32>,
    ) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();

        // Lazily size the persistent cache to the model's context (once per server lifetime).
        if kv.is_none() {
            let max_ctx = std::env::var("INFR_MAX_CTX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(self.cfg.n_ctx_train);
            *kv = Some(self.new_kv(max_ctx)?);
            cached.clear();
        }

        let gen_result = {
            let k = kv.as_mut().unwrap();
            if ids.is_empty() {
                bail!("empty prompt");
            }
            if ids.len() + 1 > k.max_ctx {
                bail!(
                    "prompt {} tokens exceeds KV capacity {} (raise INFR_MAX_CTX)",
                    ids.len(),
                    k.max_ctx
                );
            }
            // Shared-prefix reuse: rewind the cache to the common prefix, prefill only the suffix.
            // Disabled (or the prompt won't fit alongside the reply) → prefill the whole prompt.
            let no_reuse = std::env::var("INFR_NO_KV_REUSE").is_ok();
            // Always leave ≥1 token to prefill (even a fully-cached prompt) so decode has fresh logits
            // to sample the first token from — otherwise `prefill(&[])` yields no logits.
            let p = if no_reuse {
                0
            } else {
                common_prefix_len(cached, &ids).min(ids.len() - 1)
            };
            let max_new = max_new.min(k.max_ctx.saturating_sub(ids.len() + 1));
            k.len = p;
            self.prefill(&ids[p..], k)
                .and_then(|logits| self.decode_loop(logits, k, max_new, constraint, on_token))
        };

        match gen_result {
            Ok(generated) => {
                // The cache now holds exactly `ids + generated` — record that for the next diff.
                *cached = ids;
                cached.extend_from_slice(&generated);
                Ok(generated)
            }
            Err(e) => {
                // Never leave `cached` out of sync with the KV; drop the cache and start fresh next time.
                *kv = None;
                cached.clear();
                Err(e)
            }
        }
    }

    /// Build a grammar constraint that FORCES a syntactically-valid, schema-conforming tool call, for
    /// `tool_choice` values that require one (`"required"`, or `{"function":{"name":..}}` / a bare
    /// function name selecting a single tool). Returns `None` for `"auto"` / `"none"` / absent (the
    /// model decides freely). `tools` is the request's OpenAI `tools` array.
    pub fn tool_constraint(
        &self,
        tools: Option<&serde_json::Value>,
        tool_choice: Option<&str>,
    ) -> Result<Option<crate::grammar::Constraint>> {
        let Some(tools) = tools else {
            return Ok(None);
        };
        // Narrow `tools` to the chosen function when tool_choice names one; require a call only for
        // "required" or a named choice.
        crate::grammar::tool_constraint_for(
            &self.tokenizer,
            self.cfg.vocab,
            &self.cfg.eos_ids,
            Some(tools),
            tool_choice,
        )
    }

    /// Detokenize ids. `skip_special=false` preserves marker tokens (`<think>`, …) for parsing.
    pub fn decode_ids(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Greedy generate up to `max_new` tokens after `prompt` (already a chat-formatted string).
    /// One-shot: uses a fresh KV cache. For multi-turn context use [`Llama::chat_session`].
    pub fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        on_token: impl FnMut(&str),
    ) -> Result<String> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let mut kv = self.new_kv(prompt_tokens.len() + max_new + 8)?;
        let generated = self.run_in_cache(&prompt_tokens, &mut kv, max_new, on_token)?;
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Greedy generation on the backend-agnostic CPU reference path (no GPU). Mirrors
    /// [`generate`](Self::generate)'s tokenize/decode so the two are directly comparable (the
    /// CPU-vs-GPU parity tests). Returns just the text; for timing use [`CpuModel::generate_cpu`].
    pub fn generate_cpu(&self, prompt: &str, max_new: usize) -> Result<String> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        let (generated, _stats) = crate::cpu_backend::generate_dense_cpu(
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

    /// True for MoE models (qwen3moe) — use [`generate_moe`](Self::generate_moe), not the
    /// KV-resident path (which is dense-only).
    pub fn is_moe(&self) -> bool {
        self.cfg.moe.is_some()
    }

    /// Fresh MoE generation state with a GPU KV cache sized for `max_ctx` tokens.
    pub fn new_moe_kv(&self, max_ctx: usize) -> Result<MoeKv> {
        Ok(MoeKv {
            kv: self.new_kv(max_ctx)?,
            pool: None,
            dec: Some(self.build_decode_scratch(max_ctx)?),
            pf: None,
            rec_decode: None,
        })
    }

    /// Allocate the persistent decode scratch (Tier 0). Split-K attention buffers are sized for the
    /// worst-case chunk count (`chunk` is clamped to ≥64, so `n_chunks ≤ ceil(max_ctx/64)`).
    fn build_decode_scratch(&self, max_ctx: usize) -> Result<DecodeScratch> {
        let c = &self.cfg;
        let mc = c.moe.expect("moe");
        let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv, c.head_dim);
        let nblk = ne / 32;
        let ncm = max_ctx.div_ceil(64); // worst-case split-K chunk count
        let af = |n: usize| {
            self.be
                .alloc((n * 4).max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };
        let ab = |bytes: usize| {
            self.be
                .alloc(bytes.max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };
        Ok(DecodeScratch {
            hidden: af(ne)?,
            hn: af(ne)?,
            hn2: af(ne)?,
            ao: af(ne)?,
            qr: af(nh * hd)?,
            kr: af(nkv * hd)?,
            vr: af(nkv * hd)?,
            q_f16: ab(nh * hd * 2)?,
            attn: af(nh * hd)?,
            g: af(mc.n_used * mc.n_ff_exp)?,
            u: af(mc.n_used * mc.n_ff_exp)?,
            act: af(mc.n_used * mc.n_ff_exp)?,
            y: af(mc.n_used * ne)?,
            logits: af(mc.n_expert)?,
            ids: af(mc.n_used)?,
            wts: af(mc.n_used)?,
            qa: ab(ne)?,
            dact: ab(nblk * 2)?,
            sact: ab(nblk * 2)?,
            pm: af(nh * ncm)?,
            pl: af(nh * ncm)?,
            pacc: af(nh * ncm * hd)?,
            // host-visible so the host can write [pos, kv_len] per token through the mapped pointer
            params: self
                .be
                .alloc(8, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?,
            emb_in: self
                .be
                .alloc(ne * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?,
            normed: af(ne)?,
            final_logits: af(c.vocab)?,
            tok: self
                .be
                .alloc(4, BufferUsage::Readback)
                .map_err(|e| anyhow!("{e}"))?,
        })
    }

    /// Ensure `kv.pf` holds a prefill pool sized for a chunk of `t` tokens (rebuild if absent or too
    /// small — chunk size is usually constant, so this allocates once per generation).
    fn ensure_prefill_scratch(&self, kv: &mut MoeKv, t: usize) -> Result<()> {
        let m_pad = t.div_ceil(64) * 64;
        if kv.pf.as_ref().is_none_or(|p| p.m_pad < m_pad) {
            kv.pf = Some(self.build_prefill_scratch(m_pad)?);
        }
        Ok(())
    }

    /// Allocate the prefill expert scratch (one reusable set sized for `m_pad` rows).
    fn build_prefill_scratch(&self, m_pad: usize) -> Result<PrefillScratch> {
        let c = &self.cfg;
        let mc = c.moe.expect("moe");
        let (ne, nff) = (c.n_embd, mc.n_ff_exp);
        let af = |n: usize| {
            self.be
                .alloc((n * 4).max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };
        let ab = |bytes: usize| {
            self.be
                .alloc(bytes.max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };
        Ok(PrefillScratch {
            m_pad,
            xe: af(m_pad * ne)?,
            ge: af(m_pad * nff)?,
            ue: af(m_pad * nff)?,
            ae: af(m_pad * nff)?,
            ye: af(m_pad * ne)?,
            gqa: ab(m_pad * ne)?,
            gda: ab(m_pad * (ne / 32) * 2)?,
            gsa: ab(m_pad * (ne / 32) * 2)?,
            dqa: ab(m_pad * nff)?,
            dda: ab(m_pad * (nff / 32) * 2)?,
            dsa: ab(m_pad * (nff / 32) * 2)?,
        })
    }

    /// GPU attention for one MoE layer: upload the raw Q/K/V projections, then record QK-norm + RoPE
    /// (Q → f16, K → the f16 KV cache at `pos`), V → cache, and causal GQA over the cache — reusing
    /// the dense path's kernels. Returns the attention output `[n, nh*hd]` (host f32).
    #[allow(clippy::too_many_arguments)]
    fn moe_attention(
        &self,
        layer: &LayerWeights,
        q_raw: &[f32],
        k_raw: &[f32],
        v_raw: &[f32],
        kv: &KvCache,
        li: usize,
        n: usize,
        pos: usize,
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.n_head, c.n_kv, c.head_dim);
        let kvrow = nkv * hd;
        let up = |data: &[f32]| -> Result<Box<dyn Buffer>> {
            let b = self
                .be
                .alloc(data.len() * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            self.be
                .upload(b.as_ref(), bytemuck::cast_slice(data))
                .map_err(|e| anyhow!("{e}"))?;
            Ok(b)
        };
        let qr = up(q_raw)?;
        let kr = up(k_raw)?;
        let vr = up(v_raw)?;
        let q_f16 = self
            .be
            .alloc(n * nh * hd * 2, BufferUsage::Activations)
            .map_err(|e| anyhow!("{e}"))?;
        let attn = self
            .be
            .alloc(n * nh * hd * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        let (qn, kn) = (
            layer.q_norm_buf.as_ref().unwrap().as_ref(),
            layer.k_norm_buf.as_ref().unwrap().as_ref(),
        );
        rec.qk_norm_rope(
            qr.as_ref(),
            qn,
            q_f16.as_ref(),
            n,
            nh,
            hd,
            c.rope_dim,
            c.rope_theta,
            pos,
            0,
            c.rms_eps,
            None,
        );
        rec.qk_norm_rope(
            kr.as_ref(),
            kn,
            kv.k[li].as_ref(),
            n,
            nkv,
            hd,
            c.rope_dim,
            c.rope_theta,
            pos,
            pos,
            c.rms_eps,
            None,
        );
        rec.store_f16(vr.as_ref(), kv.v[li].as_ref(), n * kvrow, pos * kvrow);
        // Single-token decode (n==1) at depth: split each head's KV range across many workgroups
        // (flash-decode split-K, partials in pm/pl/pacc) so attention isn't stuck on `nh` workgroups
        // grinding the whole cache serially — the dense path's decode kernel. Prefill (n>1) uses the
        // basic per-(token,head) attention_kv. ~32 chunks/head saturates pass-1's KV bandwidth.
        let kv_len = pos + n;
        let chunk = (kv_len / 32).clamp(64, 512);
        if n == 1 && kv_len > chunk {
            let n_chunks = kv_len.div_ceil(chunk);
            let al = |elems: usize| -> Result<Box<dyn Buffer>> {
                self.be
                    .alloc((elems * 4).max(4), BufferUsage::Activations)
                    .map_err(|e| anyhow!("{e}"))
            };
            let pm = al(nh * n_chunks)?;
            let pl = al(nh * n_chunks)?;
            let pacc = al(nh * n_chunks * hd)?;
            rec.attention_kv_split(
                q_f16.as_ref(),
                kv.k[li].as_ref(),
                kv.v[li].as_ref(),
                attn.as_ref(),
                pm.as_ref(),
                pl.as_ref(),
                pacc.as_ref(),
                kv_len,
                nh,
                nkv,
                hd,
                chunk,
                n_chunks,
                0.0,
                0,
            );
        } else {
            rec.attention_kv(
                q_f16.as_ref(),
                kv.k[li].as_ref(),
                kv.v[li].as_ref(),
                attn.as_ref(),
                n,
                kv_len,
                nh,
                nkv,
                hd,
                pos,
                0,   // full causal (llama/qwen3)
                0.0, // default 1/√hd scale
            );
        }
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        let mut out = vec![0u8; n * nh * hd * 4];
        self.be
            .download(attn.as_ref(), &mut out)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(bytemuck::cast_slice(&out).to_vec())
    }

    /// Eager native GEMV `y = x·Wᵀ` against an already-resident GPU weight buffer (a streaming
    /// `ExpertPool` slot holding raw native blocks), one submit. Like `gemv_wt` but the weight is a
    /// borrowed buffer + dtype rather than an owned `Wt`.
    fn gemv_native_one(
        &self,
        w: &dyn Buffer,
        dtype: infr_core::DType,
        x: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>> {
        let xb = self
            .be
            .alloc((x.len()).max(1) * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(xb.as_ref(), bytemuck::cast_slice(x))
            .map_err(|e| anyhow!("{e}"))?;
        let yb = self
            .be
            .alloc((rows * out_f).max(1) * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        rec.linear_native(dtype, w, xb.as_ref(), yb.as_ref(), rows, in_f, out_f);
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        let mut out = vec![0u8; rows * out_f * 4];
        self.be
            .download(yb.as_ref(), &mut out)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(bytemuck::cast_slice(&out).to_vec())
    }

    /// Final norm + lm head from a single resident hidden row `src` [n_embd]. With a sampling spec
    /// the token is chosen on the GPU — argmax for greedy, or temp/top-k/top-p sampling — and only
    /// the 4-byte token id reads back; without one (or for an unsupported top_k) the full vocab
    /// logits read back for host sampling.
    fn lm_head_out(&self, src: &dyn Buffer, sample: Option<SampleParams>) -> Result<GenOut> {
        let c = &self.cfg;
        let al = |n: usize| {
            self.be
                .alloc((n * 4).max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };
        let (normed, final_logits) = (al(c.n_embd)?, al(c.vocab)?);
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        rec.rmsnorm(
            src,
            self.output_norm_buf.as_ref(),
            normed.as_ref(),
            1,
            c.n_embd,
            c.rms_eps,
        );
        rec.label_next("vocab");
        rec_linear(
            &rec,
            &self.lm_head,
            normed.as_ref(),
            final_logits.as_ref(),
            1,
            c.n_embd,
            c.vocab,
        );
        let tok = self
            .be
            .alloc(4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        // GPU-sample when possible: greedy → argmax; temp/top-k/top-p (2 ≤ top_k ≤ KMAX) → sample.
        let gpu_tok = match sample {
            Some(sp) if sp.greedy() => {
                rec.argmax(final_logits.as_ref(), tok.as_ref(), c.vocab);
                true
            }
            Some(sp) if sp.gpu_capable() => {
                rec.sample(
                    final_logits.as_ref(),
                    tok.as_ref(),
                    c.vocab,
                    sp.top_k,
                    sp.temp,
                    sp.top_p,
                    sp.u,
                );
                true
            }
            _ => false,
        };
        if gpu_tok {
            rec.finish().map_err(|e| anyhow!("{e}"))?;
            let mut tb = [0u8; 4];
            self.be
                .download(tok.as_ref(), &mut tb)
                .map_err(|e| anyhow!("{e}"))?;
            Ok(GenOut::Token(u32::from_ne_bytes(tb)))
        } else {
            rec.finish().map_err(|e| anyhow!("{e}"))?;
            let mut out = vec![0u8; c.vocab * 4];
            self.be
                .download(final_logits.as_ref(), &mut out)
                .map_err(|e| anyhow!("{e}"))?;
            Ok(GenOut::Logits(bytemuck::cast_slice(&out).to_vec()))
        }
    }

    /// GPU-resident single-token decode (qwen3moe, all experts on GPU): the residual stream stays in
    /// VRAM the whole layer — rmsnorm / QKV / attention / O / residual / ffn-norm / router are one
    /// recorder, then (after reading back only the router logits for top-k) the selected experts'
    /// gate/up/SiLU/down + weighted accumulate (`hidden += w_e·y_e`) are a second recorder. Only the
    /// `n_expert` logits cross the PCIe bus per layer — no per-matmul host round-trip. When `greedy`,
    /// samples on the GPU and returns just the token; else returns the vocab logits.
    fn forward_moe_chunk_gpu(
        &self,
        token: u32,
        kv: &mut MoeKv,
        sample: Option<SampleParams>,
    ) -> Result<GenOut> {
        let c = &self.cfg;
        let mc = c.moe.expect("moe");
        let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv, c.head_dim);
        let kvrow = nkv * hd;
        let pos = kv.kv.len;
        let kv_len = pos + 1;
        // Tier 0: persistent decode scratch — reused every token (no per-token alloc/free). Bound as
        // `&Box<dyn Buffer>` so the existing `.as_ref()` call sites are unchanged.
        let dec = kv
            .dec
            .as_ref()
            .expect("decode scratch (built in new_moe_kv)");
        let hidden = &dec.hidden;
        // Write this token's embedding row into the host-visible `emb_in` (mapped, no submit); the
        // recorded buffer copies emb_in→hidden at its start, so embedding upload isn't a per-token GPU
        // submit any more.
        let emb = &self.token_embd[token as usize * ne..(token as usize + 1) * ne];
        self.be
            .upload(dec.emb_in.as_ref(), bytemuck::cast_slice(emb))
            .map_err(|e| anyhow!("{e}"))?;
        let (hn, hn2, ao) = (&dec.hn, &dec.hn2, &dec.ao);
        let (qr, kr, vr) = (&dec.qr, &dec.kr, &dec.vr);
        let q_f16 = &dec.q_f16;
        let attn = &dec.attn;
        let (g, u, act, y) = (&dec.g, &dec.u, &dec.act, &dec.y);
        let logits = &dec.logits;
        let params = &dec.params;
        // Per-token [pos, kv_len] for the `_dyn` kernels — the only thing (besides the embedding) the
        // host writes per token, so the recorded command buffer can be replayed.
        self.be
            .upload(
                params.as_ref(),
                bytemuck::cast_slice(&[pos as u32, kv_len as u32]),
            )
            .map_err(|e| anyhow!("{e}"))?;
        // GPU-resident routing when the expert format has an id-indexed GEMV: top-k + expert ids and
        // weights stay in VRAM (one submit/layer). Else fall back to host top-k (two submits/layer).
        let (gate_dtype, _) = native_parts(&self.layers[0].moe_stacked().expect("stacked").1.gate);
        let gpu_route =
            infr_vulkan::Recorder::native_id_supported(gate_dtype) && mc.n_expert <= 128;
        let (ids_buf, wts_buf) = if gpu_route {
            (Some(&dec.ids), Some(&dec.wts))
        } else {
            (None, None)
        };
        // Q4_K experts → mmq (dp4a): quantize the ffn-normed row to int8 once (shared by gate+up).
        let mmq = gpu_route && matches!(gate_dtype, infr_core::DType::Q4K);
        let (qa, dact, sact) = if mmq {
            (Some(&dec.qa), Some(&dec.dact), Some(&dec.sact))
        } else {
            (None, None, None)
        };
        // split-K decode attention scratch (parallelize the KV reduction at depth)
        let chunk = (kv_len / 32).clamp(64, 512);
        let use_split = kv_len > chunk;
        let n_chunks = if use_split { kv_len.div_ceil(chunk) } else { 0 };
        let (pm, pl, pacc) = if use_split {
            (Some(&dec.pm), Some(&dec.pl), Some(&dec.pacc))
        } else {
            (None, None, None)
        };

        // Tier 1: the GPU-resident (gpu_route) path records ALL 48 layers into ONE command buffer and
        // submits once — vs a recorder + `queue_submit`/`queue_wait_idle` (a full GPU drain) per layer.
        // Inter-layer hazards on the shared scratch are serialized by the recorder's barrier tracking,
        // so a single submit is correct. The host-topk fallback still finishes per layer (it needs a
        // mid-layer logits readback), swapping in a fresh recorder via `mem::replace`.
        //
        // Record-once (Stage 2): when gpu_route (and not profiling), the whole forward is recorded into
        // a resubmittable command buffer keyed by the attention structure `sig`. On a cache hit we skip
        // recording entirely and just replay (only the params SSBO + embedding changed this token).
        let prof2_env = std::env::var("INFR_PROF2").is_ok();
        // Greedy decode folds the lm-head + argmax into the replayed buffer (fully record-once: one
        // replay + a 4-byte token readback). Stochastic/host sampling keeps lm_head_out separate.
        let fold_lm = matches!(sample, Some(sp) if sp.greedy());
        let sig = (use_split, chunk, n_chunks, fold_lm);
        let use_ro = gpu_route && !prof2_env;
        let hit = use_ro && kv.rec_decode.as_ref().is_some_and(|(s, _)| *s == sig);
        if !hit {
            let mut rec = if use_ro {
                self.be.recorder_persistent().map_err(|e| anyhow!("{e}"))?
            } else {
                self.be.recorder().map_err(|e| anyhow!("{e}"))?
            };
            // Copy this token's embedding (host-written into emb_in) into the GPU-only hidden, as the
            // first recorded op — so the embedding stops being a separate per-token GPU submit.
            rec.copy(dec.emb_in.as_ref(), 0, hidden.as_ref(), 0, ne * 4);
            for (li, layer) in self.layers.iter().enumerate() {
                rec.rmsnorm(
                    hidden.as_ref(),
                    layer.attn_norm_buf.as_ref(),
                    hn.as_ref(),
                    1,
                    ne,
                    c.rms_eps,
                );
                rec_linear(&rec, &layer.wq, hn.as_ref(), qr.as_ref(), 1, ne, nh * hd);
                rec_linear(&rec, &layer.wk, hn.as_ref(), kr.as_ref(), 1, ne, nkv * hd);
                rec_linear(&rec, layer.wv(), hn.as_ref(), vr.as_ref(), 1, ne, nkv * hd);
                let (qn, kn) = (
                    layer.q_norm_buf.as_ref().unwrap().as_ref(),
                    layer.k_norm_buf.as_ref().unwrap().as_ref(),
                );
                // `_dyn` kernels read pos/kv_len from `dec.params` (written once per token above), so the
                // recorded command buffer is pos-independent and can be replayed across tokens.
                rec.qk_norm_rope_dyn(
                    qr.as_ref(),
                    qn,
                    params.as_ref(),
                    None,
                    q_f16.as_ref(),
                    1,
                    nh,
                    hd,
                    c.rope_dim,
                    c.rope_theta,
                    0, // Q: out_base = 0
                    c.rms_eps,
                );
                rec.qk_norm_rope_dyn(
                    kr.as_ref(),
                    kn,
                    params.as_ref(),
                    None,
                    kv.kv.k[li].as_ref(),
                    1,
                    nkv,
                    hd,
                    c.rope_dim,
                    c.rope_theta,
                    1, // K: out_base = pos
                    c.rms_eps,
                );
                rec.store_f16_dyn(vr.as_ref(), params.as_ref(), kv.kv.v[li].as_ref(), kvrow);
                if use_split {
                    rec.attention_kv_split_dyn(
                        q_f16.as_ref(),
                        kv.kv.k[li].as_ref(),
                        kv.kv.v[li].as_ref(),
                        attn.as_ref(),
                        pm.as_ref().unwrap().as_ref(),
                        pl.as_ref().unwrap().as_ref(),
                        pacc.as_ref().unwrap().as_ref(),
                        params.as_ref(),
                        nh,
                        nkv,
                        hd,
                        chunk,
                        n_chunks,
                        0.0,
                        0,
                    );
                } else {
                    rec.attention_kv_dyn(
                        q_f16.as_ref(),
                        kv.kv.k[li].as_ref(),
                        kv.kv.v[li].as_ref(),
                        params.as_ref(),
                        attn.as_ref(),
                        1,
                        nh,
                        nkv,
                        hd,
                        0.0,
                        0,
                    );
                }
                rec_linear(&rec, &layer.wo, attn.as_ref(), ao.as_ref(), 1, nh * hd, ne);
                rec.add(hidden.as_ref(), ao.as_ref(), hidden.as_ref(), ne); // residual
                rec.rmsnorm(
                    hidden.as_ref(),
                    layer.ffn_norm_buf.as_ref(),
                    hn2.as_ref(),
                    1,
                    ne,
                    c.rms_eps,
                );
                let (gate_inp, st) = layer.moe_stacked().expect("stacked experts");
                rec_linear(
                    &rec,
                    gate_inp,
                    hn2.as_ref(),
                    logits.as_ref(),
                    1,
                    ne,
                    mc.n_expert,
                );

                if let (Some(ids), Some(wts)) = (&ids_buf, &wts_buf) {
                    // Fully GPU-resident: top-k on the GPU writes expert ids + weights to VRAM, then the
                    // selected experts' FFN (id-indexed gather of the stacked weights) accumulates into
                    // hidden — all in this one recorder. No readback, one submit/layer.
                    rec.moe_topk(
                        logits.as_ref(),
                        ids.as_ref(),
                        wts.as_ref(),
                        1,
                        mc.n_expert,
                        mc.n_used,
                        mc.scale,
                    );
                    // Fused: all n_used experts per role in ONE dispatch (concurrent, no inter-expert
                    // barrier). gate/up read the shared ffn-normed row; down reads each slot's activation.
                    let (gd, gb) = native_parts(&st.gate);
                    let (ud, ub) = native_parts(&st.up);
                    let (dd, db) = native_parts(&st.down);
                    let nu = mc.n_used;
                    if let (Some(qa), Some(da), Some(sa)) = (&qa, &dact, &sact) {
                        // Q4_K gate/up via dp4a (mmq): quantize the ffn-normed row to int8 once, shared.
                        rec.quant_q8(hn2.as_ref(), qa.as_ref(), da.as_ref(), sa.as_ref(), 1, ne);
                        rec.linear_mmv_id_multi_q4k(
                            gb,
                            qa.as_ref(),
                            da.as_ref(),
                            sa.as_ref(),
                            ids.as_ref(),
                            nu,
                            st.stride,
                            g.as_ref(),
                            ne,
                            mc.n_ff_exp,
                        );
                        rec.linear_mmv_id_multi_q4k(
                            ub,
                            qa.as_ref(),
                            da.as_ref(),
                            sa.as_ref(),
                            ids.as_ref(),
                            nu,
                            st.stride,
                            u.as_ref(),
                            ne,
                            mc.n_ff_exp,
                        );
                    } else {
                        rec.linear_native_id_multi(
                            gd,
                            gb,
                            ids.as_ref(),
                            nu,
                            st.stride,
                            hn2.as_ref(),
                            false,
                            g.as_ref(),
                            ne,
                            mc.n_ff_exp,
                        );
                        rec.linear_native_id_multi(
                            ud,
                            ub,
                            ids.as_ref(),
                            nu,
                            st.stride,
                            hn2.as_ref(),
                            false,
                            u.as_ref(),
                            ne,
                            mc.n_ff_exp,
                        );
                    }
                    rec.silu_mul(g.as_ref(), u.as_ref(), act.as_ref(), nu * mc.n_ff_exp);
                    rec.linear_native_id_multi(
                        dd,
                        db,
                        ids.as_ref(),
                        nu,
                        st.stride,
                        act.as_ref(),
                        true,
                        y.as_ref(),
                        mc.n_ff_exp,
                        ne,
                    );
                    rec.moe_accumulate(y.as_ref(), wts.as_ref(), hidden.as_ref(), ne, nu);
                    // Tier 1: do NOT finish — keep recording the next layer into the same buffer.
                } else {
                    // Fallback (non-id-capable expert format): host top-k needs this layer's logits, so
                    // finish here and continue the next layer in a fresh recorder.
                    let done = std::mem::replace(
                        &mut rec,
                        self.be.recorder().map_err(|e| anyhow!("{e}"))?,
                    );
                    done.finish().map_err(|e| anyhow!("{e}"))?;
                    let mut lb = vec![0u8; mc.n_expert * 4];
                    self.be
                        .download(logits.as_ref(), &mut lb)
                        .map_err(|e| anyhow!("{e}"))?;
                    let (idx, weights) = moe_topk(bytemuck::cast_slice(&lb), &mc);
                    let rec2 = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
                    for (ki, &e) in idx.iter().enumerate() {
                        rec_linear_expert(
                            &rec2,
                            &st.gate,
                            e,
                            st.stride,
                            hn2.as_ref(),
                            g.as_ref(),
                            1,
                            ne,
                            mc.n_ff_exp,
                        );
                        rec_linear_expert(
                            &rec2,
                            &st.up,
                            e,
                            st.stride,
                            hn2.as_ref(),
                            u.as_ref(),
                            1,
                            ne,
                            mc.n_ff_exp,
                        );
                        rec2.silu_mul(g.as_ref(), u.as_ref(), act.as_ref(), mc.n_ff_exp);
                        rec_linear_expert(
                            &rec2,
                            &st.down,
                            e,
                            st.stride,
                            act.as_ref(),
                            y.as_ref(),
                            1,
                            mc.n_ff_exp,
                            ne,
                        );
                        rec2.add_scaled(y.as_ref(), hidden.as_ref(), weights[ki], ne);
                    }
                    rec2.finish().map_err(|e| anyhow!("{e}"))?;
                }
            }
            // Greedy: fold final norm + vocab GEMV + argmax into the same (replayed) buffer, so the
            // whole token is one replay producing dec.tok. Stochastic/host: lm_head_out runs separately.
            if use_ro && fold_lm {
                rec.rmsnorm(
                    hidden.as_ref(),
                    self.output_norm_buf.as_ref(),
                    dec.normed.as_ref(),
                    1,
                    ne,
                    c.rms_eps,
                );
                rec.label_next("vocab");
                rec_linear(
                    &rec,
                    &self.lm_head,
                    dec.normed.as_ref(),
                    dec.final_logits.as_ref(),
                    1,
                    ne,
                    c.vocab,
                );
                rec.argmax(dec.final_logits.as_ref(), dec.tok.as_ref(), c.vocab);
            }
            // use_ro: keep the recorded buffer to replay; else submit it once (Tier 1) now.
            if use_ro {
                kv.rec_decode = Some((sig, rec.finish_record().map_err(|e| anyhow!("{e}"))?));
            } else {
                rec.finish().map_err(|e| anyhow!("{e}"))?;
            }
        }
        // Record-once: replay the cached command buffer (only params + embedding changed this token).
        if use_ro {
            kv.rec_decode
                .as_ref()
                .unwrap()
                .1
                .replay()
                .map_err(|e| anyhow!("{e}"))?;
        }
        kv.kv.len += 1;

        if use_ro && fold_lm {
            // Greedy fully record-once: the replayed buffer already wrote the token; just read it back.
            let mut tb = [0u8; 4];
            self.be
                .download(dec.tok.as_ref(), &mut tb)
                .map_err(|e| anyhow!("{e}"))?;
            Ok(GenOut::Token(u32::from_ne_bytes(tb)))
        } else {
            // Stochastic/host (or non-record-once): final norm + lm head + sample separately.
            self.lm_head_out(hidden.as_ref(), sample)
        }
    }

    /// GPU-resident grouped prefill (qwen3moe, all experts on GPU): like [`forward_moe_chunk_gpu`]
    /// but for a multi-token chunk. The residual stream stays in VRAM; recorder #1 does
    /// rmsnorm → QKV → attention → O → residual → ffn-norm → router for all `t` tokens; only the
    /// `t*n_expert` router logits read back for host top-k. Recorder #2 runs the FFN grouped by
    /// expert — for each active expert: gather its token rows on the GPU, one SwiGLU GEMM, then a
    /// weighted scatter-add back into the resident hidden. No per-expert host round-trip. Returns
    /// last-token logits.
    fn forward_moe_chunk_gpu_prefill(
        &self,
        tokens: &[u32],
        kv: &mut MoeKv,
        sample: Option<SampleParams>,
    ) -> Result<GenOut> {
        let c = &self.cfg;
        let mc = c.moe.expect("moe");
        let t = tokens.len();
        let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv, c.head_dim);
        let nff = mc.n_ff_exp;
        let kvrow = nkv * hd;
        let pos = kv.kv.len;
        let kv_len = pos + t;
        let al = |n: usize| {
            self.be
                .alloc((n * 4).max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };
        let ab = |bytes: usize| {
            self.be
                .alloc(bytes.max(4), BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))
        };

        // resident scratch (reused across all layers)
        let hidden = al(t * ne)?;
        let mut emb = vec![0f32; t * ne];
        for (i, &tok) in tokens.iter().enumerate() {
            emb[i * ne..(i + 1) * ne]
                .copy_from_slice(&self.token_embd[tok as usize * ne..(tok as usize + 1) * ne]);
        }
        self.be
            .upload(hidden.as_ref(), bytemuck::cast_slice(&emb))
            .map_err(|e| anyhow!("{e}"))?;
        // Projections (QKV/O) run as tiled GEMMs → outputs are M-padded to gmp = ceil(t/64)*64.
        let gmp = t.div_ceil(64) * 64;
        let (hn, hn2) = (al(t * ne)?, al(t * ne)?);
        let ao = al(gmp * ne)?;
        let (qr, kr, vr) = (al(gmp * nh * hd)?, al(gmp * nkv * hd)?, al(gmp * nkv * hd)?);
        // Q4_K Q/K/O projections use dp4a (mmq): quantize the projection inputs (hn for Q/K, attn for
        // O) to int8 once each. q4_proj gates on Q (q/k/o are Q4_K in this model; v is Q6_K → coopmat).
        let q4_proj = matches!(native_parts(&self.layers[0].wq).0, infr_core::DType::Q4K);
        let qbufs = |in_f: usize| -> Result<QBufs> {
            Ok((
                ab(gmp * in_f)?,
                ab(gmp * (in_f / 32) * 2)?,
                ab(gmp * (in_f / 32) * 2)?,
            ))
        };
        let (qa_h, da_h, sa_h, qa_o, da_o, sa_o) = if q4_proj {
            let (a, b, c2) = qbufs(ne)?;
            let (d, e, f) = qbufs(nh * hd)?;
            (Some(a), Some(b), Some(c2), Some(d), Some(e), Some(f))
        } else {
            (None, None, None, None, None, None)
        };
        // Flash prefill attention (split-K, register-blocked, never materializes the score matrix) is
        // hd=128-specialized and wants 64-row tiles → pad q/attn to mpad rows. Small chunks (t<64) or
        // other head dims fall back to the basic per-query attention_kv. INFR_NO_FLASH forces fallback.
        // bm=32 flash tile needs 29056 B shared; skip flash on devices that can't fit it (see the
        // dense-prefill use_flash above) so the pipeline never over-commits shared memory.
        let use_flash = hd == 128
            && t >= 64
            && self.be.max_shared_memory_bytes() >= 32 * infr_vulkan::FLASH_SHARED_PER_ROW
            && std::env::var("INFR_NO_FLASH").is_err();
        let mpad = if use_flash { t.div_ceil(64) * 64 } else { t };
        let q_f16 = self
            .be
            .alloc(mpad * nh * hd * 2, BufferUsage::Activations)
            .map_err(|e| anyhow!("{e}"))?;
        let attn = al(mpad * nh * hd)?;
        // Flash split-K scratch: po=[≤8,mpad,nh,hd] partials, pm/pl=[≤8,mpad,nh] (reused across layers).
        let flash = if use_flash {
            Some((
                al(8 * mpad * nh * hd)?,
                al(8 * mpad * nh)?,
                al(8 * mpad * nh)?,
            ))
        } else {
            None
        };
        let logits = al(t * mc.n_expert)?;
        // GPU routing (n_expert ≤ 128 for the top-k workgroup): per-token top-k → bucket tokens by
        // expert entirely on the GPU. Only the per-expert counts/offsets (n_expert u32 each) read
        // back, to size the per-expert GEMM dispatches. Else fall back to host top-k + index uploads.
        let gpu_route = mc.n_expert <= 128;
        let n_pairs = t * mc.n_used;
        let rb = |n: usize| {
            self.be
                .alloc((n * 4).max(4), BufferUsage::Readback)
                .map_err(|e| anyhow!("{e}"))
        };
        let route = if gpu_route {
            Some((
                al(n_pairs)?,     // tok_ids
                al(n_pairs)?,     // tok_wts
                rb(mc.n_expert)?, // counts (downloaded)
                rb(mc.n_expert)?, // offsets (downloaded + used on GPU by scatter)
                al(mc.n_expert)?, // fill
                al(n_pairs)?,     // bucket_rows
                al(n_pairs)?,     // bucket_wts
            ))
        } else {
            None
        };

        // Persistent expert pool (reused across all layers + chunks) so the FFN doesn't churn ~8
        // buffer allocs per active expert per layer.
        self.ensure_prefill_scratch(kv, t)?;

        for (li, layer) in self.layers.iter().enumerate() {
            // recorder 1: attention + router for all t tokens, on the GPU.
            let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
            rec.rmsnorm(
                hidden.as_ref(),
                layer.attn_norm_buf.as_ref(),
                hn.as_ref(),
                t,
                ne,
                c.rms_eps,
            );
            if let (Some(qa), Some(da), Some(sa)) = (&qa_h, &da_h, &sa_h) {
                // Q4_K Q/K via dp4a (quantize hn once); V (Q6_K) via coopmat.
                rec.quant_q8(hn.as_ref(), qa.as_ref(), da.as_ref(), sa.as_ref(), t, ne);
                let (_, wqb) = native_parts(&layer.wq);
                let (_, wkb) = native_parts(&layer.wk);
                rec.matmul_mmq_q4k(
                    qa.as_ref(),
                    da.as_ref(),
                    sa.as_ref(),
                    wqb,
                    0,
                    qr.as_ref(),
                    t,
                    ne,
                    nh * hd,
                );
                rec.matmul_mmq_q4k(
                    qa.as_ref(),
                    da.as_ref(),
                    sa.as_ref(),
                    wkb,
                    0,
                    kr.as_ref(),
                    t,
                    ne,
                    nkv * hd,
                );
            } else {
                rec_proj(&rec, &layer.wq, hn.as_ref(), qr.as_ref(), t, ne, nh * hd);
                rec_proj(&rec, &layer.wk, hn.as_ref(), kr.as_ref(), t, ne, nkv * hd);
            }
            rec_proj(&rec, layer.wv(), hn.as_ref(), vr.as_ref(), t, ne, nkv * hd);
            let (qn, kn) = (
                layer.q_norm_buf.as_ref().unwrap().as_ref(),
                layer.k_norm_buf.as_ref().unwrap().as_ref(),
            );
            rec.qk_norm_rope(
                qr.as_ref(),
                qn,
                q_f16.as_ref(),
                t,
                nh,
                hd,
                c.rope_dim,
                c.rope_theta,
                pos,
                0,
                c.rms_eps,
                None,
            );
            rec.qk_norm_rope(
                kr.as_ref(),
                kn,
                kv.kv.k[li].as_ref(),
                t,
                nkv,
                hd,
                c.rope_dim,
                c.rope_theta,
                pos,
                pos,
                c.rms_eps,
                None,
            );
            rec.store_f16(vr.as_ref(), kv.kv.v[li].as_ref(), t * kvrow, pos * kvrow);
            if let Some((po, pm, pl)) = &flash {
                rec.attention_prefill_flash(
                    q_f16.as_ref(),
                    kv.kv.k[li].as_ref(),
                    kv.kv.v[li].as_ref(),
                    attn.as_ref(),
                    po.as_ref(),
                    pm.as_ref(),
                    pl.as_ref(),
                    t,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                );
            } else {
                rec.attention_kv(
                    q_f16.as_ref(),
                    kv.kv.k[li].as_ref(),
                    kv.kv.v[li].as_ref(),
                    attn.as_ref(),
                    t,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                    0,   // full causal (MoE attention)
                    0.0, // default 1/√hd scale
                );
            }
            if let (Some(qa), Some(da), Some(sa)) = (&qa_o, &da_o, &sa_o) {
                rec.quant_q8(
                    attn.as_ref(),
                    qa.as_ref(),
                    da.as_ref(),
                    sa.as_ref(),
                    t,
                    nh * hd,
                );
                let (_, wob) = native_parts(&layer.wo);
                rec.matmul_mmq_q4k(
                    qa.as_ref(),
                    da.as_ref(),
                    sa.as_ref(),
                    wob,
                    0,
                    ao.as_ref(),
                    t,
                    nh * hd,
                    ne,
                );
            } else {
                rec_proj(&rec, &layer.wo, attn.as_ref(), ao.as_ref(), t, nh * hd, ne);
            }
            rec.add(hidden.as_ref(), ao.as_ref(), hidden.as_ref(), t * ne); // residual
            rec.rmsnorm(
                hidden.as_ref(),
                layer.ffn_norm_buf.as_ref(),
                hn2.as_ref(),
                t,
                ne,
                c.rms_eps,
            );
            let (gate_inp, st) = layer.moe_stacked().expect("stacked experts");
            rec_linear(
                &rec,
                gate_inp,
                hn2.as_ref(),
                logits.as_ref(),
                t,
                ne,
                mc.n_expert,
            );

            #[allow(clippy::type_complexity, unused_assignments)]
            let mut fallback_bufs: Option<(Box<dyn Buffer>, Box<dyn Buffer>)> = None;
            let (counts_h, offs_h, bucket_rows, bucket_wts) =
                if let Some((tok_ids, tok_wts, counts, offsets, fill, bucket_rows, bucket_wts)) =
                    &route
                {
                    // GPU routing: per-token top-k → count/scan/scatter buckets, all on the GPU.
                    rec.moe_topk(
                        logits.as_ref(),
                        tok_ids.as_ref(),
                        tok_wts.as_ref(),
                        t,
                        mc.n_expert,
                        mc.n_used,
                        mc.scale,
                    );
                    rec.zero(counts.as_ref(), mc.n_expert);
                    rec.moe_bucket_count(tok_ids.as_ref(), counts.as_ref(), n_pairs);
                    rec.moe_bucket_scan(
                        counts.as_ref(),
                        offsets.as_ref(),
                        fill.as_ref(),
                        mc.n_expert,
                    );
                    rec.moe_bucket_scatter(
                        tok_ids.as_ref(),
                        tok_wts.as_ref(),
                        offsets.as_ref(),
                        fill.as_ref(),
                        bucket_rows.as_ref(),
                        bucket_wts.as_ref(),
                        n_pairs,
                        mc.n_used,
                    );
                    rec.finish().map_err(|e| anyhow!("{e}"))?;
                    // Read back only the per-expert counts + offsets (n_expert u32 each) to size dispatches.
                    let mut cb = vec![0u8; mc.n_expert * 4];
                    let mut ob = vec![0u8; mc.n_expert * 4];
                    self.be
                        .download(counts.as_ref(), &mut cb)
                        .map_err(|e| anyhow!("{e}"))?;
                    self.be
                        .download(offsets.as_ref(), &mut ob)
                        .map_err(|e| anyhow!("{e}"))?;
                    (
                        bytemuck::cast_slice::<u8, u32>(&cb).to_vec(),
                        bytemuck::cast_slice::<u8, u32>(&ob).to_vec(),
                        Some(bucket_rows),
                        Some(bucket_wts),
                    )
                } else {
                    // Fallback: host top-k → per-expert index buffers uploaded to GPU.
                    rec.finish().map_err(|e| anyhow!("{e}"))?;
                    let mut lb = vec![0u8; t * mc.n_expert * 4];
                    self.be
                        .download(logits.as_ref(), &mut lb)
                        .map_err(|e| anyhow!("{e}"))?;
                    let lh: &[f32] = bytemuck::cast_slice(&lb);
                    let mut rows_of: Vec<Vec<u32>> = vec![Vec::new(); mc.n_expert];
                    let mut wts_of: Vec<Vec<f32>> = vec![Vec::new(); mc.n_expert];
                    for r in 0..t {
                        let (idx, w) = moe_topk(&lh[r * mc.n_expert..(r + 1) * mc.n_expert], &mc);
                        for (ki, &e) in idx.iter().enumerate() {
                            rows_of[e].push(r as u32);
                            wts_of[e].push(w[ki]);
                        }
                    }
                    // Concatenate into the shared bucket layout (offsets = prefix sum) and upload once.
                    let mut offs = vec![0u32; mc.n_expert];
                    let mut acc = 0u32;
                    for e in 0..mc.n_expert {
                        offs[e] = acc;
                        acc += rows_of[e].len() as u32;
                    }
                    let mut rows_flat = Vec::with_capacity(n_pairs);
                    let mut wts_flat = Vec::with_capacity(n_pairs);
                    for e in 0..mc.n_expert {
                        rows_flat.extend_from_slice(&rows_of[e]);
                        wts_flat.extend_from_slice(&wts_of[e]);
                    }
                    let br = al(rows_flat.len().max(1))?;
                    let bw = al(wts_flat.len().max(1))?;
                    self.be
                        .upload(br.as_ref(), bytemuck::cast_slice(&rows_flat))
                        .map_err(|e| anyhow!("{e}"))?;
                    self.be
                        .upload(bw.as_ref(), bytemuck::cast_slice(&wts_flat))
                        .map_err(|e| anyhow!("{e}"))?;
                    let counts: Vec<u32> =
                        (0..mc.n_expert).map(|e| rows_of[e].len() as u32).collect();
                    fallback_bufs = Some((br, bw));
                    let (br, bw) = fallback_bufs.as_ref().unwrap();
                    (counts, offs, Some(br), Some(bw))
                };

            // recorder 2: per active expert, gather its bucket slice → SwiGLU GEMM → weighted
            // scatter-add into hidden. m/offset come from the GPU-built (or host) routing.
            let (bucket_rows, bucket_wts) = (bucket_rows.unwrap(), bucket_wts.unwrap());
            let rec2 = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
            let pool = kv.pf.as_ref().expect("prefill scratch (built above)");
            for e in 0..mc.n_expert {
                let m = counts_h[e] as usize;
                if m == 0 {
                    continue;
                }
                let off = offs_h[e] as usize;
                // One reusable scratch set: each expert reuses it (serializing via the recorder's
                // barriers — a K-sweep showed experts serialize anyway, so the win is removing the
                // per-expert alloc churn, not concurrency).
                let s = pool;
                let (xe, ge, ue, ae, ye) = (&s.xe, &s.ge, &s.ue, &s.ae, &s.ye);
                rec2.gather_rows(hn2.as_ref(), bucket_rows.as_ref(), off, xe.as_ref(), m, ne);
                // gate/up: Q4_K → dp4a (mmq) GEMM (int8 dot, faster than coopmat-f16); quantize the
                // gathered batch to int8 once, shared by both. down (Q6_K) stays on the coopmat GEMM.
                if matches!(native_parts(&st.gate).0, infr_core::DType::Q4K) {
                    rec2.quant_q8(
                        xe.as_ref(),
                        s.gqa.as_ref(),
                        s.gda.as_ref(),
                        s.gsa.as_ref(),
                        m,
                        ne,
                    );
                    let (_, gb) = native_parts(&st.gate);
                    let (_, ub) = native_parts(&st.up);
                    let base = e * st.stride;
                    rec2.label_next("expert_gateup");
                    rec2.matmul_mmq_q4k(
                        s.gqa.as_ref(),
                        s.gda.as_ref(),
                        s.gsa.as_ref(),
                        gb,
                        base,
                        ge.as_ref(),
                        m,
                        ne,
                        nff,
                    );
                    rec2.label_next("expert_gateup");
                    rec2.matmul_mmq_q4k(
                        s.gqa.as_ref(),
                        s.gda.as_ref(),
                        s.gsa.as_ref(),
                        ub,
                        base,
                        ue.as_ref(),
                        m,
                        ne,
                        nff,
                    );
                } else {
                    rec2.label_next("expert_gateup");
                    rec_gemm_expert(
                        &rec2,
                        &st.gate,
                        e,
                        st.stride,
                        xe.as_ref(),
                        ge.as_ref(),
                        m,
                        ne,
                        nff,
                    );
                    rec2.label_next("expert_gateup");
                    rec_gemm_expert(
                        &rec2,
                        &st.up,
                        e,
                        st.stride,
                        xe.as_ref(),
                        ue.as_ref(),
                        m,
                        ne,
                        nff,
                    );
                }
                rec2.silu_mul(ge.as_ref(), ue.as_ref(), ae.as_ref(), m * nff);
                // down: Q6_K → dp4a (mmq) GEMM (int8 dot, faster than coopmat-f16); quantize the
                // SwiGLU activations to int8 per 32-block first. Else coopmat-f16 fallback.
                if matches!(native_parts(&st.down).0, infr_core::DType::Q6K) {
                    rec2.quant_q8(
                        ae.as_ref(),
                        s.dqa.as_ref(),
                        s.dda.as_ref(),
                        s.dsa.as_ref(),
                        m,
                        nff,
                    );
                    let (_, db) = native_parts(&st.down);
                    rec2.label_next("expert_down");
                    rec2.matmul_mmq_q6k(
                        s.dqa.as_ref(),
                        s.dda.as_ref(),
                        db,
                        e * st.stride,
                        ye.as_ref(),
                        m,
                        nff,
                        ne,
                    );
                } else {
                    rec2.label_next("expert_down");
                    rec_gemm_expert(
                        &rec2,
                        &st.down,
                        e,
                        st.stride,
                        ae.as_ref(),
                        ye.as_ref(),
                        m,
                        nff,
                        ne,
                    );
                }
                rec2.scatter_add_rows(
                    ye.as_ref(),
                    bucket_rows.as_ref(),
                    bucket_wts.as_ref(),
                    off,
                    hidden.as_ref(),
                    m,
                    ne,
                );
            }
            rec2.finish().map_err(|e| anyhow!("{e}"))?;
        }
        kv.kv.len += t;

        // Gather hidden's last row on the GPU, then final norm + lm head (+ greedy GPU argmax).
        let last_idx = al(1)?;
        self.be
            .upload(last_idx.as_ref(), bytemuck::cast_slice(&[(t - 1) as u32]))
            .map_err(|e| anyhow!("{e}"))?;
        let hlast = al(ne)?;
        let rec = self.be.recorder().map_err(|e| anyhow!("{e}"))?;
        rec.gather_rows(hidden.as_ref(), last_idx.as_ref(), 0, hlast.as_ref(), 1, ne);
        rec.finish().map_err(|e| anyhow!("{e}"))?;
        self.lm_head_out(hlast.as_ref(), sample)
    }

    /// Eager MoE forward for one chunk of `tokens` at positions `kv.pos..`, appending K/V to the
    /// cache (so decode steps process only the new token, not the whole sequence). Returns logits
    /// (`vocab`) for the last token. Same math as [`forward_moe`] but cached.
    pub fn forward_moe_chunk(&self, tokens: &[u32], kv: &mut MoeKv) -> Result<Vec<f32>> {
        match self.forward_moe_chunk_g(tokens, kv, None)? {
            GenOut::Logits(l) => Ok(l),
            GenOut::Token(_) => unreachable!("no sampler always returns logits"),
        }
    }

    /// As [`forward_moe_chunk`] but with on-GPU greedy sampling: when `greedy`, the GPU argmaxes the
    /// vocab logits and only the 4-byte token id crosses the bus (no vocab-logits download).
    fn forward_moe_chunk_g(
        &self,
        tokens: &[u32],
        kv: &mut MoeKv,
        sample: Option<SampleParams>,
    ) -> Result<GenOut> {
        // Stacked GPU expert bank → fully GPU-resident path (no per-matmul host round-trip):
        // single-token decode, or grouped-by-expert prefill for a multi-token chunk. Offloaded /
        // per-expert layers use the eager path.
        if self.layers[0].moe_stacked().is_some() {
            return if tokens.len() == 1 {
                self.forward_moe_chunk_gpu(tokens[0], kv, sample)
            } else {
                self.forward_moe_chunk_gpu_prefill(tokens, kv, sample)
            };
        }
        let c = &self.cfg;
        let mc = c.moe.expect("forward_moe_chunk requires a MoE model");
        let t = tokens.len();
        let (ne, nh, nkv, hd) = (c.n_embd, c.n_head, c.n_kv, c.head_dim);
        let pos0 = kv.kv.len;

        let mut hidden = vec![0f32; t * ne];
        for (i, &tok) in tokens.iter().enumerate() {
            hidden[i * ne..(i + 1) * ne]
                .copy_from_slice(&self.token_embd[tok as usize * ne..(tok as usize + 1) * ne]);
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // attention with GPU KV cache — Q/K/V projections batched into one submit, then QK-norm /
            // RoPE / KV-append / attention on the GPU (reusing the dense kernels via moe_attention).
            let hn = rmsnorm_rows(&hidden, &layer.attn_norm, t, ne, c.rms_eps);
            let mut qkv = self.gemv_wt_many(&[
                (&layer.wq, hn.as_slice(), t, ne, nh * hd),
                (&layer.wk, hn.as_slice(), t, ne, nkv * hd),
                (layer.wv(), hn.as_slice(), t, ne, nkv * hd),
            ])?;
            let vnew = qkv.pop().unwrap();
            let knew = qkv.pop().unwrap();
            let q = qkv.pop().unwrap();
            let attn = self.moe_attention(layer, &q, &knew, &vnew, &kv.kv, li, t, pos0)?;
            let ao = self.gemv_wt(&layer.wo, &attn, t, nh * hd, ne)?;
            for i in 0..t * ne {
                hidden[i] += ao[i];
            }

            // MoE FFN: route each token to top-k experts, weighted SwiGLU sum
            let hn2 = rmsnorm_rows(&hidden, &layer.ffn_norm, t, ne, c.rms_eps);
            let (gate_inp, experts) = layer.moe();
            let logits = self.gemv_wt(gate_inp, &hn2, t, ne, mc.n_expert)?;
            if !experts[0].gate.is_cpu() {
                // All experts GPU-resident → group tokens by expert and run one SwiGLU GEMM per
                // expert (tiled coopmat) instead of `t × n_used` per-token GEMVs.
                let ffn = self.moe_ffn_grouped(&hn2, &logits, experts, &mc, t)?;
                for i in 0..t * ne {
                    hidden[i] += ffn[i];
                }
            } else {
                // Host-offloaded / streamed experts: per-token path (CPU or VRAM pool).
                for r in 0..t {
                    let out_row = self.moe_ffn_token(
                        &hn2[r * ne..(r + 1) * ne],
                        &logits[r * mc.n_expert..(r + 1) * mc.n_expert],
                        experts,
                        &mc,
                        li,
                        &mut kv.pool,
                    )?;
                    for i in 0..ne {
                        hidden[r * ne + i] += out_row[i];
                    }
                }
            }
        }
        kv.kv.len += t;

        // Eager (offloaded) path always returns logits; the caller samples on the host.
        let _ = sample;
        let last = &hidden[(t - 1) * ne..t * ne];
        let normed = rmsnorm_rows(last, &self.output_norm, 1, ne, c.rms_eps);
        Ok(GenOut::Logits(self.gemv_wt(
            &self.lm_head,
            &normed,
            1,
            ne,
            c.vocab,
        )?))
    }

    /// Raw quantized bytes of a host-backed expert's `role` weight ("gate"/"up"/"down"), read
    /// zero-copy from the GGUF mmap. Each expert is a contiguous `1/n_expert` slice of the stacked
    /// `ffn_{role}_exps` tensor.
    fn expert_bytes(&self, li: usize, role: &str, e: usize) -> Result<&[u8]> {
        let name = format!("blk.{li}.ffn_{role}_exps.weight");
        let all = self
            .gguf
            .tensor_bytes(&name)
            .map_err(|er| anyhow!("{er}"))?;
        let n_expert = self.cfg.moe.expect("moe").n_expert;
        let stride = all.len() / n_expert;
        Ok(&all[e * stride..(e + 1) * stride])
    }

    /// (dtype, mmap bytes) for a host-backed expert role — the inputs to a CPU/stream matmul.
    fn host_expert(
        &self,
        ew: &ExpertW,
        li: usize,
        role: &str,
        e: usize,
    ) -> Result<(infr_core::DType, &[u8])> {
        let ExpertW::Cpu { dtype } = ew else {
            unreachable!("host_expert on a GPU expert");
        };
        Ok((*dtype, self.expert_bytes(li, role, e)?))
    }

    /// One token's MoE FFN: softmax router → renormalized top-k → weighted SwiGLU sum over the
    /// selected experts. `x` is the (already ffn-normed) token `[n_embd]`, `rl` its router logits.
    /// `li` = layer index (for streaming-pool keys); `pool` = the streaming VRAM pool (lazily built).
    fn moe_ffn_token(
        &self,
        x: &[f32],
        rl: &[f32],
        experts: &[ExpertWt],
        mc: &MoeConfig,
        li: usize,
        pool: &mut Option<infr_vulkan::ExpertPool>,
    ) -> Result<Vec<f32>> {
        let ne = self.cfg.n_embd;
        let maxl = rl.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = rl.iter().map(|&v| (v - maxl).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for pr in probs.iter_mut() {
            *pr /= sum;
        }
        let mut idx: Vec<usize> = (0..mc.n_expert).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        idx.truncate(mc.n_used);
        let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);

        // Each expert's SwiGLU → `ys[ki]` (down output). Expert placement is per-layer uniform:
        // host-offloaded layers (`INFR_NCMOE`) run on the CPU, or — with `INFR_MOE_STREAM` and a
        // native-supported quant — stream the active experts into a VRAM pool and GPU-compute them;
        // otherwise the experts are GPU-resident and batched.
        let host_layer = !idx.is_empty() && experts[idx[0]].gate.is_cpu();
        let stream_layer = host_layer
            && self.moe_stream
            && matches!(&experts[idx[0]].gate, ExpertW::Cpu { dtype } if is_native_default(*dtype));
        let ys: Vec<Vec<f32>> = if stream_layer {
            self.stream_experts(x, &idx, experts, mc, li, pool)?
        } else if host_layer {
            idx.iter()
                .map(|&e| {
                    let (gdt, gb) = self.host_expert(&experts[e].gate, li, "gate", e)?;
                    let gate = cpu_expert_matvec(gdt, gb, x, ne, mc.n_ff_exp)?;
                    let (udt, ub) = self.host_expert(&experts[e].up, li, "up", e)?;
                    let up = cpu_expert_matvec(udt, ub, x, ne, mc.n_ff_exp)?;
                    let act: Vec<f32> = (0..mc.n_ff_exp).map(|i| silu(gate[i]) * up[i]).collect();
                    let (ddt, db) = self.host_expert(&experts[e].down, li, "down", e)?;
                    cpu_expert_matvec(ddt, db, &act, mc.n_ff_exp, ne)
                })
                .collect::<Result<_>>()?
        } else {
            // Phase 1: all gate+up matmuls in ONE submit (they all read `x`).
            let mut gu_ops: Vec<(&Wt, &[f32], usize, usize, usize)> =
                Vec::with_capacity(idx.len() * 2);
            for &e in &idx {
                gu_ops.push((experts[e].gate.gpu(), x, 1, ne, mc.n_ff_exp));
                gu_ops.push((experts[e].up.gpu(), x, 1, ne, mc.n_ff_exp));
            }
            let gu = self.gemv_wt_many(&gu_ops)?;
            let acts: Vec<Vec<f32>> = (0..idx.len())
                .map(|ki| {
                    let (gate, up) = (&gu[2 * ki], &gu[2 * ki + 1]);
                    (0..mc.n_ff_exp).map(|i| silu(gate[i]) * up[i]).collect()
                })
                .collect();
            // Phase 2: all down matmuls in ONE submit.
            let down_ops: Vec<(&Wt, &[f32], usize, usize, usize)> = idx
                .iter()
                .enumerate()
                .map(|(ki, &e)| {
                    (
                        experts[e].down.gpu(),
                        acts[ki].as_slice(),
                        1,
                        mc.n_ff_exp,
                        ne,
                    )
                })
                .collect();
            self.gemv_wt_many(&down_ops)?
        };

        // Host weighted accumulate over the renormalized top-k softmax weights.
        let mut out = vec![0f32; ne];
        for (ki, &e) in idx.iter().enumerate() {
            let w_e = probs[e] / wsum * mc.scale;
            for i in 0..ne {
                out[i] += w_e * ys[ki][i];
            }
        }
        Ok(out)
    }

    /// Group-by-expert MoE FFN over a whole chunk of `t` tokens (all experts GPU-resident — the
    /// prefill path). Routes every token to its top-k experts on the host, then for each expert
    /// gathers all of its assigned token rows into one contiguous batch and runs **one** SwiGLU
    /// per expert as a tiled GEMM (`[m_e×ne]·Wᵀ`) — gate+up batched into a single submit, down into
    /// a second — instead of `t × n_used` per-token GEMVs. Scatter-adds the weighted expert outputs
    /// back to each token's row. Returns the `[t*ne]` FFN output to add into the residual stream.
    fn moe_ffn_grouped(
        &self,
        hn2: &[f32],    // [t*ne], ffn-normed token rows
        logits: &[f32], // [t*n_expert], router logits
        experts: &[ExpertWt],
        mc: &MoeConfig,
        t: usize,
    ) -> Result<Vec<f32>> {
        let ne = self.cfg.n_embd;
        let nff = mc.n_ff_exp;

        // Route: per expert, the token rows it must process and their renormalized weights.
        let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); mc.n_expert];
        let mut wts_of: Vec<Vec<f32>> = vec![Vec::new(); mc.n_expert];
        for r in 0..t {
            let (idx, weights) = moe_topk(&logits[r * mc.n_expert..(r + 1) * mc.n_expert], mc);
            for (ki, &e) in idx.iter().enumerate() {
                rows_of[e].push(r);
                wts_of[e].push(weights[ki]);
            }
        }
        let active: Vec<usize> = (0..mc.n_expert)
            .filter(|&e| !rows_of[e].is_empty())
            .collect();

        // Gather each active expert's token rows into a contiguous [m_e*ne] batch.
        let xs: Vec<Vec<f32>> = active
            .iter()
            .map(|&e| {
                let mut x = vec![0f32; rows_of[e].len() * ne];
                for (j, &r) in rows_of[e].iter().enumerate() {
                    x[j * ne..(j + 1) * ne].copy_from_slice(&hn2[r * ne..(r + 1) * ne]);
                }
                x
            })
            .collect();

        // Phase 1: every active expert's gate+up GEMM in ONE submit (both read its batch `xs[ai]`).
        let mut gu_ops: Vec<(&Wt, &[f32], usize, usize, usize)> =
            Vec::with_capacity(active.len() * 2);
        for (ai, &e) in active.iter().enumerate() {
            let m = rows_of[e].len();
            gu_ops.push((experts[e].gate.gpu(), xs[ai].as_slice(), m, ne, nff));
            gu_ops.push((experts[e].up.gpu(), xs[ai].as_slice(), m, ne, nff));
        }
        let gu = self.gemv_wt_many(&gu_ops)?;

        // SwiGLU on host, then Phase 2: every active expert's down GEMM in ONE submit.
        let acts: Vec<Vec<f32>> = (0..active.len())
            .map(|ai| {
                let (g, u) = (&gu[2 * ai], &gu[2 * ai + 1]);
                (0..g.len()).map(|i| silu(g[i]) * u[i]).collect()
            })
            .collect();
        let down_ops: Vec<(&Wt, &[f32], usize, usize, usize)> = active
            .iter()
            .enumerate()
            .map(|(ai, &e)| {
                (
                    experts[e].down.gpu(),
                    acts[ai].as_slice(),
                    rows_of[e].len(),
                    nff,
                    ne,
                )
            })
            .collect();
        let ys = self.gemv_wt_many(&down_ops)?;

        // Scatter-add each expert's weighted down output back to its token rows.
        let mut out = vec![0f32; t * ne];
        for (ai, &e) in active.iter().enumerate() {
            let y = &ys[ai];
            for (j, &r) in rows_of[e].iter().enumerate() {
                let w = wts_of[e][j];
                for i in 0..ne {
                    out[r * ne + i] += w * y[j * ne + i];
                }
            }
        }
        Ok(out)
    }

    /// Stream a host-offloaded layer's active experts through the VRAM `ExpertPool` and GPU-compute
    /// them (`INFR_MOE_STREAM`): for each selected expert, make its gate/up/down resident in a pool
    /// slot (upload-on-miss, LRU-evict) and run the native GEMV against the slot. Returns each
    /// expert's down output. Faster than the CPU path (GPU matmul), VRAM bounded to the pool.
    fn stream_experts(
        &self,
        x: &[f32],
        idx: &[usize],
        experts: &[ExpertWt],
        mc: &MoeConfig,
        li: usize,
        pool: &mut Option<infr_vulkan::ExpertPool>,
    ) -> Result<Vec<Vec<f32>>> {
        use infr_vulkan::linear::pad_to_u32_align;
        let ne = self.cfg.n_embd;
        // (dtype, native-padded mmap bytes) for an expert role — bytes read zero-copy then padded.
        let parts = |ew: &ExpertW, role: &str, ex: usize| -> Result<(infr_core::DType, Vec<u8>)> {
            let (dt, b) = self.host_expert(ew, li, role, ex)?;
            Ok((dt, pad_to_u32_align(b)))
        };
        // Lazily size the pool: one slot per expert-role's native-padded bytes, enough for a layer's
        // active set (n_used × 3 roles) plus headroom — bounded VRAM regardless of expert count.
        if pool.is_none() {
            let stride = parts(&experts[idx[0]].gate, "gate", idx[0])?
                .1
                .len()
                .max(parts(&experts[idx[0]].down, "down", idx[0])?.1.len());
            let n_slots = (mc.n_used * 3 + mc.n_used).max(8);
            *pool = Some(
                infr_vulkan::ExpertPool::new(&self.be, stride, n_slots)
                    .map_err(|e| anyhow!("{e}"))?,
            );
        }
        let pool = pool.as_mut().unwrap();
        let mut ys = Vec::with_capacity(idx.len());
        for &ex in idx {
            let key = |role: usize| li * mc.n_expert * 3 + ex * 3 + role;
            let (gdt, gb) = parts(&experts[ex].gate, "gate", ex)?;
            let gbuf = pool
                .resident(&self.be, key(0), &gb)
                .map_err(|e| anyhow!("{e}"))?;
            let gate = self.gemv_native_one(gbuf, gdt, x, 1, ne, mc.n_ff_exp)?;
            let (udt, ub) = parts(&experts[ex].up, "up", ex)?;
            let ubuf = pool
                .resident(&self.be, key(1), &ub)
                .map_err(|e| anyhow!("{e}"))?;
            let up = self.gemv_native_one(ubuf, udt, x, 1, ne, mc.n_ff_exp)?;
            let act: Vec<f32> = (0..mc.n_ff_exp).map(|i| silu(gate[i]) * up[i]).collect();
            let (ddt, db) = parts(&experts[ex].down, "down", ex)?;
            let dbuf = pool
                .resident(&self.be, key(2), &db)
                .map_err(|e| anyhow!("{e}"))?;
            ys.push(self.gemv_native_one(dbuf, ddt, &act, 1, mc.n_ff_exp, ne)?);
        }
        Ok(ys)
    }

    /// MoE generation (qwen3moe) with a host KV cache — prefill the prompt once, then decode one
    /// token per step (no O(n²) recompute). `prompt` is chat-formatted; `on_token` fires per token.
    pub fn generate_moe(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let tokens: Vec<u32> = enc.get_ids().to_vec();
        let sampler = self.sampler.get();
        let mut rng = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9e3779b97f4a7c15)
            | 1;
        // Sample on the GPU when possible (only the 4-byte token id reads back); the forward falls
        // back to returning logits for configs the GPU sampler can't handle, which we sample here.
        let sp = |rng: &mut u64| {
            Some(SampleParams {
                temp: sampler.temp,
                top_k: sampler.top_k,
                top_p: sampler.top_p,
                u: draw_u(rng),
            })
        };
        let resolve = |out: GenOut, rng: &mut u64| match out {
            GenOut::Token(t) => t,
            GenOut::Logits(l) => sample_logits(&l, sampler, rng),
        };
        let mut kv = self.new_moe_kv(tokens.len() + max_new + 8)?;
        let s = sp(&mut rng);
        let mut out = self.forward_moe_chunk_g(&tokens, &mut kv, s)?; // prefill
        let mut stream = StreamDecoder::default();
        let mut generated: Vec<u32> = Vec::new();
        for _ in 0..max_new {
            let next = resolve(out, &mut rng);
            if self.cfg.eos_ids.contains(&next) {
                break;
            }
            generated.push(next);
            let full = self.tokenizer.decode(&generated, true).unwrap_or_default();
            on_token(&stream.step(&full));
            let s = sp(&mut rng);
            out = self.forward_moe_chunk_g(&[next], &mut kv, s)?; // 1-token decode
        }
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Start a stateful multi-turn chat with a KV cache sized for `max_ctx` tokens. Each turn keeps
    /// prior context resident, so only the new tokens are prefilled.
    pub fn chat_session(&self, max_ctx: usize) -> Result<ChatSession<'_>> {
        Ok(ChatSession {
            llama: self,
            kv: self.new_kv(max_ctx)?,
            started: false,
            last_prompt_tokens: 0,
            cached: Vec::new(),
        })
    }
}

/// A stateful multi-turn chat over a persistent KV cache (so the model sees prior turns). Create via
/// [`Llama::chat_session`]. The shared [`crate::model::Chat`] owns the conversation history and
/// `<think>`-stripping; this session only owns the KV cache: it [`render`](Self::render)s the
/// caller-supplied message list through the model's template and [`generate`](Self::generate)s the
/// reply, prefilling only the token suffix that differs from what's already cached.
pub struct ChatSession<'a> {
    llama: &'a Llama,
    kv: KvCache,
    started: bool,
    pub(crate) last_prompt_tokens: usize,
    /// The token sequence currently materialized in the KV cache, so each turn can prefill only the
    /// new suffix (common-prefix diff vs the freshly-rendered prompt).
    cached: Vec<u32>,
}

impl ChatSession<'_> {
    /// Tokens of context currently held (all prior turns + their replies).
    pub fn ctx_len(&self) -> usize {
        self.kv.len
    }

    /// Prompt tokens prefilled in the most recent [`turn`](Self::turn) (the ChatML-wrapped user
    /// message, including any turn-open markers). Use for prefill-rate stats.
    pub fn last_prompt_tokens(&self) -> usize {
        self.last_prompt_tokens
    }

    /// KV-cache capacity in tokens.
    pub fn max_ctx(&self) -> usize {
        self.kv.max_ctx
    }

    /// Render `messages` `(role, content)` through the model's OWN embedded chat template — the
    /// [`crate::model::ChatModel::render`] primitive for the dense/Gemma GPU path. Errors if the GGUF
    /// has no `tokenizer.chat_template` or it fails to render (no hardcoded fallback).
    pub fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.llama
            .render_chat_messages(messages, true)
            .ok_or_else(no_template_err)
    }

    /// Generate the assistant reply for an already-rendered `prompt`, streaming decoded pieces to
    /// `on_token`. Keeps the persistent KV cache warm: only the token suffix that differs from the
    /// cached prefix is prefilled (incremental prefill), so a multi-turn REPL re-prefills just the
    /// new turn's markers + the model's prior answer, not the whole conversation. Returns per-turn
    /// [`GenStats`] (`n_prompt` = the suffix actually prefilled this turn).
    pub fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        mut on_token: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let t0 = std::time::Instant::now();
        let ids = self.llama.encode_special(prompt)?;
        if std::env::var("INFR_DEBUG_TOKENS").is_ok() {
            let dump: Vec<(u32, String)> = ids
                .iter()
                .map(|&id| (id, self.llama.tokenizer.id_to_token(id).unwrap_or_default()))
                .collect();
            eprintln!("[tokens] {dump:?}");
        }
        // Prefill only the suffix that differs from what's already in the cache.
        let p = common_prefix_len(&self.cached, &ids);
        let new = &ids[p..];
        let n_new = new.len();
        let room = self.kv.max_ctx.saturating_sub(p + n_new + 1);
        if room == 0 {
            bail!(
                "context full: {} prompt vs {} cap — start a new session",
                ids.len(),
                self.kv.max_ctx
            );
        }
        let max_new = max_new.min(room);
        self.started = true;
        self.kv.len = p; // rewind the cache to the shared prefix
        self.last_prompt_tokens = n_new;
        let logits = self.llama.prefill(new, &mut self.kv)?;
        // Split prefill vs decode time at the first streamed piece (matches the other backends).
        let mut t_first: Option<std::time::Instant> = None;
        let mut n_gen = 0usize;
        let generated = self
            .llama
            .decode_loop(logits, &mut self.kv, max_new, None, |piece| {
                if t_first.is_none() {
                    t_first = Some(std::time::Instant::now());
                }
                n_gen += 1;
                on_token(piece);
            })?;
        // The cache now holds the rendered prompt + the raw generation.
        self.cached = ids;
        self.cached.extend_from_slice(&generated);
        let now = std::time::Instant::now();
        let tf = t_first.unwrap_or(now);
        Ok(crate::GenStats {
            n_prompt: n_new,
            prompt_secs: tf.duration_since(t0).as_secs_f64(),
            n_gen,
            decode_secs: now.duration_since(tf).as_secs_f64(),
        })
    }
}

/// Length of the shared leading run of two token sequences (for incremental prefill).
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

struct CacheSlot {
    kv: Option<KvCache>,
    cached: Vec<u32>,
    /// Tokens of the last rendered PROMPT that was prefilled (before generation). A request
    /// CONTINUES this slot only if it shares this whole prompt — not merely the system prefix — which
    /// distinguishes a real follow-up turn from a different conversation that reuses the same system
    /// prompt but a different (possibly short) user message.
    prompt_len: usize,
    lru: u64,
}

/// Multi-slot prefix cache for `infr serve`: up to `max_slots` persistent KV caches so
/// concurrent / interleaved conversations don't thrash a single cache. Each request is routed to
/// the slot it CONTINUES (that slot's whole cached sequence is a prefix of the request, within a
/// small slack for reply re-tokenization); a genuinely new conversation takes a free slot, or evicts
/// the least-recently-used one. Slots are allocated lazily (a single client only ever uses one, so
/// VRAM is unchanged for that case). `INFR_KV_SLOTS` sets the slot count (default 4); the per-slot
/// KV is sized like the single-cache path (`INFR_MAX_CTX` / the model's trained context). Reuse math
/// per slot is the tested [`Llama::generate_ids_cached`]; this only adds the routing.
pub struct ServeCache {
    slots: Vec<CacheSlot>,
    max_slots: usize,
    tick: u64,
}

impl ServeCache {
    /// Build from the environment: `INFR_KV_SLOTS` slots (default 4, min 1).
    pub fn from_env() -> Self {
        let max_slots = std::env::var("INFR_KV_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4usize)
            .max(1);
        Self {
            slots: Vec::new(),
            max_slots,
            tick: 0,
        }
    }

    /// Generate a reply for `prompt`, routing to the right slot and prefilling only the divergent
    /// suffix (see [`Llama::generate_ids_cached`]). Returns the generated token ids.
    pub fn generate(
        &mut self,
        llama: &Llama,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        on_token: impl FnMut(&str),
    ) -> Result<Vec<u32>> {
        self.tick += 1;
        let ids: Vec<u32> = llama
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        let idx = self.select(llama, &ids)?;
        self.slots[idx].lru = self.tick;
        let slot = &mut self.slots[idx];
        let res = llama.generate_ids_cached(
            prompt,
            max_new,
            constraint,
            on_token,
            &mut slot.kv,
            &mut slot.cached,
        );
        // Record the prompt length for the continuation test; on error the cache was invalidated.
        self.slots[idx].prompt_len = if res.is_ok() { ids.len() } else { 0 };
        res
    }

    /// Pick the slot index to use. In priority: (1) an existing conversation this request CONTINUES
    /// (its whole cache is a near-prefix of the request); (2) a new conversation → a free/LRU slot,
    /// RADIX-SEEDED by copying the longest shared-prefix KV from another slot (e.g. the system prompt)
    /// so that prefix isn't re-prefilled. Returns the chosen (possibly seeded) slot index.
    fn select(&mut self, llama: &Llama, ids: &[u32]) -> Result<usize> {
        // MIN_SEED: below this, a cross-slot KV copy (recorder + submit) isn't worth vs. re-prefill.
        const MIN_SEED: usize = 64;
        if std::env::var("INFR_NO_KV_REUSE").is_ok() {
            return Ok(self.grow_or_lru(None));
        }
        // One pass: find the continuation slot AND the best radix donor (max shared prefix).
        let mut cont: Option<(usize, usize)> = None; // (idx, prompt_len) — deepest continued prompt
        let mut donor: Option<(usize, usize)> = None; // (idx, shared_prefix) — best KV to copy
        for (i, s) in self.slots.iter().enumerate() {
            if s.cached.is_empty() {
                continue;
            }
            let cp = common_prefix_len(&s.cached, ids);
            // CONTINUATION = the request shares this slot's WHOLE prompt (not just the system prefix),
            // so it's a follow-up turn. Using prompt_len (not cached.len) is robust to reply
            // re-tokenization drift AND doesn't misfire when a *different* conversation reuses the same
            // system prompt but a shorter user message.
            if cp >= s.prompt_len && s.prompt_len > 0 && cont.is_none_or(|(_, l)| s.prompt_len > l)
            {
                cont = Some((i, s.prompt_len));
            }
            let dp = cp.min(ids.len().saturating_sub(1)); // always leave ≥1 token to prefill
            if s.kv.is_some() && donor.is_none_or(|(_, p)| dp > p) {
                donor = Some((i, dp));
            }
        }
        if let Some((i, _)) = cont {
            return Ok(i);
        }
        // New conversation: pick a free/LRU slot (not the donor), then seed it from the donor's prefix.
        let target = self.grow_or_lru(donor.map(|(i, _)| i));
        if let Some((di, p)) = donor {
            if p >= MIN_SEED && di != target && std::env::var("INFR_NO_RADIX").is_err() {
                if self.slots[target].kv.is_none() {
                    let max_ctx = self.max_ctx(llama);
                    self.slots[target].kv = Some(llama.new_kv(max_ctx)?);
                }
                // Copy donor.kv[0..p] → target.kv and adopt the matching prefix as target's cache;
                // the ensuing generate then prefills only ids[p..]. (Sequential borrows: di != target.)
                let donor_kv = self.slots[di].kv.take().expect("donor has a KV");
                let prefix = self.slots[di].cached[..p].to_vec();
                llama.kv_copy_prefix(self.slots[target].kv.as_mut().unwrap(), &donor_kv, p)?;
                self.slots[di].kv = Some(donor_kv);
                self.slots[target].cached = prefix;
            }
        }
        Ok(target)
    }

    /// A free slot (grow up to `max_slots`) or the LRU one (evicted: clear its cache, keep its KV
    /// alloc → no VRAM churn). `exclude` is kept out of eviction (the radix donor we're copying from).
    fn grow_or_lru(&mut self, exclude: Option<usize>) -> usize {
        if self.slots.len() < self.max_slots {
            self.slots.push(CacheSlot {
                kv: None,
                cached: Vec::new(),
                prompt_len: 0,
                lru: self.tick,
            });
            return self.slots.len() - 1;
        }
        let lru = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i) != exclude)
            .min_by_key(|(_, s)| s.lru)
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.slots[lru].cached.clear();
        lru
    }

    fn max_ctx(&self, llama: &Llama) -> usize {
        std::env::var("INFR_MAX_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(llama.cfg.n_ctx_train)
    }
}

// ---- host ops ----

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// MoE router top-k on host: softmax the `n_expert` logits, take the `n_used` highest, renormalize
/// their probs and apply the routing `scale`. Returns (expert indices, per-expert weights).
fn moe_topk(rl: &[f32], mc: &MoeConfig) -> (Vec<usize>, Vec<f32>) {
    let maxl = rl.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = rl.iter().map(|&v| (v - maxl).exp()).collect();
    let mut idx: Vec<usize> = (0..mc.n_expert).collect();
    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    idx.truncate(mc.n_used);
    let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
    let weights: Vec<f32> = idx.iter().map(|&e| probs[e] / wsum * mc.scale).collect();
    (idx, weights)
}

fn rmsnorm_rows(x: &[f32], w: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for i in 0..dim {
            y[r * dim + i] = row[i] * scale * w[i];
        }
    }
    y
}

/// Host matvec `y = x·Wᵀ` for a host-backed expert weight: dequant the quantized `[out_f, in_f]`
/// `bytes` (read zero-copy from the GGUF mmap) to f32, then dot each row with `x`. Correctness-first
/// — the CPU path is the VRAM/speed tradeoff; not micro-optimized (full dequant per call).
fn cpu_expert_matvec(
    dtype: infr_core::DType,
    bytes: &[u8],
    x: &[f32],
    in_f: usize,
    out_f: usize,
) -> Result<Vec<f32>> {
    let w = dequant_block(dtype, bytes)?; // [out_f * in_f] row-major (out rows)
    let mut y = vec![0f32; out_f];
    for o in 0..out_f {
        let row = &w[o * in_f..(o + 1) * in_f];
        y[o] = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
    Ok(y)
}

/// ggml NORM rope (interleaved pairs (2i, 2i+1)), applied per head over the first `rope_dim` dims.
fn rope_rows(x: &mut [f32], t: usize, n_heads: usize, hd: usize, rope_dim: usize, theta: f32) {
    for pos in 0..t {
        for h in 0..n_heads {
            let base = (pos * n_heads + h) * hd;
            for i in 0..rope_dim / 2 {
                let freq = (theta as f64).powf(-2.0 * i as f64 / rope_dim as f64) as f32;
                let ang = pos as f32 * freq;
                let (s, co) = ang.sin_cos();
                let a = x[base + 2 * i];
                let b = x[base + 2 * i + 1];
                x[base + 2 * i] = a * co - b * s;
                x[base + 2 * i + 1] = a * s + b * co;
            }
        }
    }
}

/// Causal GQA attention. q [T, nh*hd], k/v [T, nkv*hd] -> out [T, nh*hd].
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
) -> Vec<f32> {
    let scale = 1.0 / (hd as f32).sqrt();
    let group = nh / nkv;
    let mut out = vec![0f32; t * nh * hd];
    for ti in 0..t {
        for h in 0..nh {
            let kvh = h / group;
            let qv = &q[(ti * nh + h) * hd..(ti * nh + h) * hd + hd];
            // scores over j in 0..=ti (causal)
            let mut scores = vec![0f32; ti + 1];
            let mut maxs = f32::NEG_INFINITY;
            for j in 0..=ti {
                let kv = &k[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                let mut dot = 0f32;
                for d in 0..hd {
                    dot += qv[d] * kv[d];
                }
                dot *= scale;
                scores[j] = dot;
                if dot > maxs {
                    maxs = dot;
                }
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - maxs).exp();
                sum += *s;
            }
            let ob = (ti * nh + h) * hd;
            for j in 0..=ti {
                let p = scores[j] / sum;
                let vv = &v[(j * nkv + kvh) * hd..(j * nkv + kvh) * hd + hd];
                for d in 0..hd {
                    out[ob + d] += p * vv[d];
                }
            }
        }
    }
    out
}

/// Validate that the native raw-block GPU GEMV (`linear_native`) matches the CPU dequant for each
/// affine quant type — the single upload path now that `Wt::Q` (host repack + `linear_q`) is gone.
#[cfg(test)]
mod gpu_affine_tests {
    use super::*;
    use infr_core::backend::BufferUsage;
    use infr_core::Backend;
    use infr_vulkan::VulkanBackend;

    // ── Native-block GPU-vs-CPU parity tests ────────────────────────────────
    //
    // Each test: build a known raw block, run `linear_native` GEMV with x=all-1.0,
    // compare to `dequant_unified`/`dequant_codebook` CPU sum (dot with 1.0 = weight sum).

    fn check_native(dtype: infr_core::DType, block_bytes: &[u8]) {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        use infr_vulkan::linear::pad_to_u32_align;

        // CPU reference: sum of dequantized weights (dot with all-1.0 input)
        let (qv, sc, mn) = dequant_unified(dtype, block_bytes);
        let numel = qv.len();
        let cpu_out: f32 = (0..numel).map(|g| sc[g] * qv[g] as f32 + mn[g]).sum();

        // Upload native raw block bytes (padded to u32)
        let padded = pad_to_u32_align(block_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let x: Vec<f32> = vec![1.0f32; numel];
        let xbuf = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(4, BufferUsage::Readback).unwrap();

        let rec = be.recorder().unwrap();
        rec.linear_native(
            dtype,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            numel,
            1,
        );
        rec.finish().unwrap();

        let mut out_bytes = vec![0u8; 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_out: f32 = bytemuck::cast_slice(&out_bytes)[0];

        let err = (gpu_out - cpu_out).abs();
        let rel = err / (cpu_out.abs() + 1e-6);
        assert!(
            rel < 5e-3,
            "{dtype:?} native GPU vs CPU: gpu={gpu_out} cpu={cpu_out} err={err} rel={rel}"
        );
    }

    // ── Phase 0: Q8_0 ────────────────────────────────────────────────────────

    #[test]
    fn q8_0_native_matches_cpu() {
        // d=1.5, qs: bytes 0..32 = signed values -128..127 cycling
        let d_bits = half::f16::from_f32(1.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&d_bits);
        for i in 0..32u8 {
            // values: 0,1,..,127,-128,-127,...,-97 → will cycle through positive and negative
            block[2 + i as usize] = i.wrapping_add(100); // e.g. 100,101,..,127,-128,...
        }
        check_native(infr_core::DType::Q8_0, &block);
    }

    // ── Phase 1: Q4_0, Q4_1, Q5_0, Q5_1 ─────────────────────────────────────

    #[test]
    fn q4_0_native_matches_cpu() {
        // d=2.0, qs all=0x89 (lo=9,hi=8) → mix of positive/negative after -8
        let d_bits = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bits);
        for b in &mut block[2..18] {
            *b = 0x89;
        }
        check_native(infr_core::DType::Q4_0, &block);
    }

    #[test]
    fn q4_1_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&m_bits);
        for b in &mut block[4..20] {
            *b = 0x31;
        }
        check_native(infr_core::DType::Q4_1, &block);
    }

    #[test]
    fn q5_0_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&d_bits);
        // qh=0 (no high bits), qs all=0x0A → q5 values 10 (lo) and 0 (hi)
        for b in &mut block[6..22] {
            *b = 0x0A;
        }
        check_native(infr_core::DType::Q5_0, &block);
    }

    #[test]
    fn q5_1_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bits = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&m_bits);
        for b in &mut block[8..24] {
            *b = 0x1F;
        }
        check_native(infr_core::DType::Q5_1, &block);
    }

    // ── Phase 2: k-quants ─────────────────────────────────────────────────────

    #[test]
    fn q2k_native_matches_cpu() {
        let mut block = vec![0u8; 84];
        block[0] = 0x03;
        block[1] = 0x03;
        for b in &mut block[16..80] {
            *b = 0x55;
        }
        block[80..82].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        check_native(infr_core::DType::Q2K, &block);
    }

    #[test]
    fn q3k_native_matches_cpu() {
        let mut block = vec![0u8; 110];
        block[108..110].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        check_native(infr_core::DType::Q3K, &block);
    }

    #[test]
    fn q4k_native_matches_cpu() {
        // d=1.0, dmin=0.5, scales[0]=0x33 → sc=3, mn=3
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let dmin_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&dmin_bits);
        // scales[4..16]: all 0x33 → k4(0)=(3,3) for first sub-block
        for b in &mut block[4..16] {
            *b = 0x33;
        }
        // qs: alternating 0xAB
        for b in &mut block[16..144] {
            *b = 0xAB;
        }
        check_native(infr_core::DType::Q4K, &block);
    }

    #[test]
    fn q5k_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let dmin_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&dmin_bits);
        for b in &mut block[4..16] {
            *b = 0x33;
        }
        for b in &mut block[48..176] {
            *b = 0xAB;
        }
        check_native(infr_core::DType::Q5K, &block);
    }

    /// Non-uniform Q5K block: distinct scales per sub-block + non-zero qh.
    /// The uniform tests above are insensitive to indexing bugs; this one is not.
    #[test]
    fn q5k_native_nonuniform() {
        // Build a block where each sub-block has a different scale and qh is varied.
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let dmin_bits = half::f16::from_f32(0.25).to_bits().to_le_bytes();
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&dmin_bits);
        // scales[0..12]: encode 8 distinct 6-bit (scale,min) pairs via k4 encoding.
        // Use simple encoding: first 4 bytes = low bits of sc (i=0..3), bytes 4..8 = low bits of mn,
        // bytes 8..12 = upper bits mixed.
        // Set them to varied values so each sub-block has a different scale.
        block[4] = 0x20; // k4(0): sc=0x20&0x3F=32, mn=block[8]&0x3F
        block[5] = 0x10; // k4(2): sc=16, mn=...
        block[6] = 0x08; // k4(4): sc computed via else branch
        block[7] = 0x04; // k4(6): sc computed via else branch
        block[8] = 0x3F; // k4(0): mn=63
        block[9] = 0x2A; // k4(2): mn=42
        block[10] = 0x15; // k4(4): (used in else branch)
        block[11] = 0x09; // k4(6): (used in else branch)
                          // block[12..16] could affect k4(4..7) upper bits; set to varied pattern
        block[12] = 0xC0; // affects k4(4): sc upper bits from (block[8]>>6)<<4 = (0x3F>>6)<<4=0
        block[13] = 0x80;
        block[14] = 0x40;
        block[15] = 0x20;
        // qh: set to varied pattern so high bits vary
        for i in 0..32usize {
            block[16 + i] = (i as u8).wrapping_mul(17).wrapping_add(1);
        }
        // qs: set to varied pattern
        for i in 0..128usize {
            block[48 + i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        check_native(infr_core::DType::Q5K, &block);
    }

    /// Non-uniform Q6K block: distinct scales per sub-block.
    #[test]
    fn q6k_native_nonuniform() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 210];
        // ql: varied
        for i in 0..128usize {
            block[i] = (i as u8).wrapping_mul(11).wrapping_add(3);
        }
        // qh: varied
        for i in 0..64usize {
            block[128 + i] = (i as u8).wrapping_mul(7).wrapping_add(5);
        }
        // scales: varied signed int8 values (avoid extreme negatives to keep sums finite)
        for i in 0..16usize {
            block[192 + i] = ((i as u8).wrapping_mul(5) + 8) & 0x7F;
        } // positive only
        block[208..210].copy_from_slice(&d_bits);
        check_native(infr_core::DType::Q6K, &block);
    }

    /// Multi-block Q5K test: 4 blocks (in_f=1024), out_f=2. Tests cross-block access.
    #[test]
    fn q5k_native_multiblock() {
        use infr_vulkan::linear::pad_to_u32_align;
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        // Build 8 distinct Q5K blocks (in_f=2048, out_f=2 → weight matrix [2, 2048])
        const N_BLOCKS: usize = 8;
        const BLOCK_SZ: usize = 176;
        const NELEMS: usize = 256;
        const IN_F: usize = N_BLOCKS * NELEMS;
        const OUT_F: usize = 2;
        // Total weight elements: OUT_F * IN_F = 2 * 2048 = 4096 = 16 blocks
        const TOTAL_BLOCKS: usize = OUT_F * IN_F / NELEMS; // = OUT_F * N_BLOCKS
        let mut w_bytes = vec![0u8; TOTAL_BLOCKS * BLOCK_SZ];
        // Fill blocks with distinct, varied data
        for b in 0..TOTAL_BLOCKS {
            let off = b * BLOCK_SZ;
            let d_bits = half::f16::from_f32(0.5 + b as f32 * 0.1)
                .to_bits()
                .to_le_bytes();
            let dmin_bits = half::f16::from_f32(0.1).to_bits().to_le_bytes();
            w_bytes[off..off + 2].copy_from_slice(&d_bits);
            w_bytes[off + 2..off + 4].copy_from_slice(&dmin_bits);
            for i in 0..12 {
                w_bytes[off + 4 + i] = ((b * 12 + i) as u8).wrapping_mul(3) | 0x20;
            }
            for i in 0..32 {
                w_bytes[off + 16 + i] = ((b * 32 + i) as u8).wrapping_mul(17);
            }
            for i in 0..128 {
                w_bytes[off + 48 + i] = ((b * 128 + i) as u8).wrapping_mul(7).wrapping_add(3);
            }
        }
        // CPU reference: compute expected outputs using dequant_unified
        let mut cpu_outputs = [0f32; OUT_F];
        let x: Vec<f32> = (0..IN_F).map(|i| 1.0f32 + i as f32 * 0.001f32).collect();
        for o in 0..OUT_F {
            let w_row_bytes = &w_bytes[o * N_BLOCKS * BLOCK_SZ..(o + 1) * N_BLOCKS * BLOCK_SZ];
            let (qv, sc, mn) = dequant_unified(infr_core::DType::Q5K, w_row_bytes);
            let sum: f32 = (0..IN_F)
                .map(|i| (sc[i] * qv[i] as f32 + mn[i]) * x[i])
                .sum();
            cpu_outputs[o] = sum;
        }
        // GPU: upload and run
        let padded = pad_to_u32_align(&w_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(IN_F * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(OUT_F * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q5K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            IN_F,
            OUT_F,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; OUT_F * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_outputs: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        for o in 0..OUT_F {
            let err = (gpu_outputs[o] - cpu_outputs[o]).abs();
            let rel = err / (cpu_outputs[o].abs() + 1e-3);
            assert!(
                rel < 5e-3,
                "Q5K out[{o}]: gpu={} cpu={} err={err} rel={rel}",
                gpu_outputs[o],
                cpu_outputs[o]
            );
        }
    }

    /// Full-scale Q6K test matching ffn_down dimensions: out_f=1024, in_f=3072.
    #[test]
    fn q6k_native_fullscale() {
        use infr_vulkan::linear::pad_to_u32_align;
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        const BLOCK_SZ: usize = 210;
        const NELEMS: usize = 256;
        const IN_F: usize = 3072;
        const OUT_F: usize = 1024;
        let n_blocks_per_row = IN_F / NELEMS; // 12
        let total_blocks = OUT_F * n_blocks_per_row;
        let mut w_bytes = vec![0u8; total_blocks * BLOCK_SZ];
        for b in 0..total_blocks {
            let off = b * BLOCK_SZ;
            let d_bits = half::f16::from_f32(0.1 + (b % 16) as f32 * 0.05)
                .to_bits()
                .to_le_bytes();
            for i in 0..128 {
                w_bytes[off + i] = ((b * 7 + i) as u8).wrapping_mul(11);
            }
            for i in 0..64 {
                w_bytes[off + 128 + i] = ((b * 3 + i) as u8).wrapping_mul(7);
            }
            for i in 0..16 {
                w_bytes[off + 192 + i] = (((b + i) as u8).wrapping_mul(5) + 8) & 0x7F;
            }
            w_bytes[off + 208..off + 210].copy_from_slice(&d_bits);
        }
        let x: Vec<f32> = (0..IN_F).map(|i| 1.0f32 + i as f32 * 0.001f32).collect();
        // Only check a few output elements to keep test fast
        let check_rows = [0usize, 1, 100, 1023];
        let padded = pad_to_u32_align(&w_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(IN_F * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(OUT_F * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q6K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            IN_F,
            OUT_F,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; OUT_F * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_outputs: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        for &o in &check_rows {
            let w_row_bytes =
                &w_bytes[o * n_blocks_per_row * BLOCK_SZ..(o + 1) * n_blocks_per_row * BLOCK_SZ];
            let (qv, sc, mn) = dequant_unified(infr_core::DType::Q6K, w_row_bytes);
            let cpu: f32 = (0..IN_F)
                .map(|i| (sc[i] * qv[i] as f32 + mn[i]) * x[i])
                .sum();
            let err = (gpu_outputs[o] - cpu).abs();
            let rel = err / (cpu.abs() + 1e-3);
            assert!(
                rel < 5e-3,
                "Q6K fullscale out[{o}]: gpu={} cpu={cpu} err={err} rel={rel}",
                gpu_outputs[o]
            );
        }
    }

    /// Multi-block Q6K test: 8 blocks, out_f=2. Tests cross-block access.
    #[test]
    fn q6k_native_multiblock() {
        use infr_vulkan::linear::pad_to_u32_align;
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        const N_BLOCKS: usize = 4;
        const BLOCK_SZ: usize = 210;
        const NELEMS: usize = 256;
        const IN_F: usize = N_BLOCKS * NELEMS;
        const OUT_F: usize = 2;
        const TOTAL_BLOCKS: usize = OUT_F * N_BLOCKS;
        let mut w_bytes = vec![0u8; TOTAL_BLOCKS * BLOCK_SZ];
        for b in 0..TOTAL_BLOCKS {
            let off = b * BLOCK_SZ;
            let d_bits = half::f16::from_f32(0.5 + b as f32 * 0.1)
                .to_bits()
                .to_le_bytes();
            for i in 0..128 {
                w_bytes[off + i] = ((b * 128 + i) as u8).wrapping_mul(11).wrapping_add(3);
            }
            for i in 0..64 {
                w_bytes[off + 128 + i] = ((b * 64 + i) as u8).wrapping_mul(7).wrapping_add(5);
            }
            for i in 0..16 {
                w_bytes[off + 192 + i] = (((b * 16 + i) as u8).wrapping_mul(5) + 8) & 0x7F;
            }
            w_bytes[off + 208..off + 210].copy_from_slice(&d_bits);
        }
        let mut cpu_outputs = [0f32; OUT_F];
        let x: Vec<f32> = (0..IN_F).map(|i| 1.0f32 + i as f32 * 0.001f32).collect();
        for o in 0..OUT_F {
            let w_row_bytes = &w_bytes[o * N_BLOCKS * BLOCK_SZ..(o + 1) * N_BLOCKS * BLOCK_SZ];
            let (qv, sc, mn) = dequant_unified(infr_core::DType::Q6K, w_row_bytes);
            let sum: f32 = (0..IN_F)
                .map(|i| (sc[i] * qv[i] as f32 + mn[i]) * x[i])
                .sum();
            cpu_outputs[o] = sum;
        }
        let padded = pad_to_u32_align(&w_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(IN_F * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(OUT_F * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q6K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            IN_F,
            OUT_F,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; OUT_F * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_outputs: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        for o in 0..OUT_F {
            let err = (gpu_outputs[o] - cpu_outputs[o]).abs();
            let rel = err / (cpu_outputs[o].abs() + 1e-3);
            assert!(
                rel < 5e-3,
                "Q6K out[{o}]: gpu={} cpu={} err={err} rel={rel}",
                gpu_outputs[o],
                cpu_outputs[o]
            );
        }
    }

    #[test]
    fn q6k_native_matches_cpu() {
        // d=0.5, scales[0..16]=0x20 (i8=32), ql=0xFF, qh=0xFF → q6=63
        let d_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 210];
        for b in &mut block[0..128] {
            *b = 0xFF;
        } // ql
        for b in &mut block[128..192] {
            *b = 0xFF;
        } // qh
        for b in &mut block[192..208] {
            *b = 0x20;
        } // scales = +32
        block[208..210].copy_from_slice(&d_bits);
        check_native(infr_core::DType::Q6K, &block);
    }

    /// Verify Q6K native shader handles f16 subnormal d values correctly.
    /// Real model weights use subnormal d (e.g. d_bits=0x0140 ≈ 1.9e-5), which
    /// naive f16→f32 that maps e=0 to 0 will silently zero out every output.
    #[test]
    fn q6k_native_subnormal_d() {
        // d_bits = 0x0140 (e=0, m=0x140=320): subnormal f16 ≈ 1.9073e-5
        let d_bits: u16 = 0x0140;
        let mut block = vec![0u8; 210];
        for b in &mut block[0..128] {
            *b = 0xFF;
        } // ql all-1
        for b in &mut block[128..192] {
            *b = 0xFF;
        } // qh all-1
        for b in &mut block[192..208] {
            *b = 0x20;
        } // scales = i8 +32
        block[208..210].copy_from_slice(&d_bits.to_le_bytes());
        check_native(infr_core::DType::Q6K, &block);
    }

    /// Load a real Q6K tensor from the model and verify GPU vs CPU.
    #[test]
    fn q6k_real_model_tensor() {
        use infr_vulkan::linear::pad_to_u32_align;
        let Some(model_path) = crate::test_qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not in the HF cache");
            return;
        };
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        let g = infr_gguf::Gguf::open(&model_path).unwrap();
        // attn_v.weight blk.0: Q6K, [1024, 1024] → in_f=1024, out_f=1024
        let tensor_name = "blk.0.attn_v.weight";
        let bytes = g.tensor_bytes(tensor_name).unwrap();
        let in_f = 1024usize;
        let out_f = 1024usize;
        // CPU ref: dot each output row against x=all-1.0
        let (qv, sc, mn) = dequant_unified(infr_core::DType::Q6K, bytes);
        let numel = in_f * out_f;
        assert_eq!(qv.len(), numel, "element count mismatch");
        let x: Vec<f32> = vec![1.0f32; in_f];
        let mut cpu_out = vec![0f32; out_f];
        for o in 0..out_f {
            cpu_out[o] = (0..in_f)
                .map(|i| sc[o * in_f + i] * qv[o * in_f + i] as f32 + mn[o * in_f + i])
                .sum();
        }
        // GPU
        let padded = pad_to_u32_align(bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(in_f * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(out_f * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q6K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            in_f,
            out_f,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; out_f * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        let mut max_err = 0f32;
        let mut max_idx = 0;
        let mut n_zero = 0usize;
        for o in 0..out_f {
            let err = (gpu_out[o] - cpu_out[o]).abs();
            if gpu_out[o] == 0.0 && cpu_out[o].abs() > 0.1 {
                n_zero += 1;
            }
            if err > max_err {
                max_err = err;
                max_idx = o;
            }
        }
        // Print first 5 failing elements
        let mut n_print = 0;
        for o in 0..out_f {
            let rel = (gpu_out[o] - cpu_out[o]).abs() / (cpu_out[o].abs() + 1e-3);
            if rel > 5e-3 && n_print < 5 {
                eprintln!("FAIL out[{o}]: gpu={} cpu={}", gpu_out[o], cpu_out[o]);
                n_print += 1;
            }
        }
        eprintln!("Real Q6K: n_zero={n_zero}/{out_f}, max_err={max_err} at out[{max_idx}]");
        let rel = max_err / (cpu_out[max_idx].abs() + 1e-3);
        assert!(
            rel < 5e-3,
            "Real Q6K tensor: max_err={max_err} at out[{max_idx}]: gpu={} cpu={} rel={rel}",
            gpu_out[max_idx],
            cpu_out[max_idx]
        );
    }

    // ── Native-block prefill GEMM parity (matmul_native vs trusted linear_native) ──
    //
    // The tiled coopmat GEMM reuses the same per-format dqblk decode as the GEMV, so the decode is
    // already covered by the *_native_matches_cpu tests. This guards the NEW code — the 64x64 tile,
    // shared staging, and coopmat accumulation — by checking that C[m,:] from matmul_native equals
    // the GEMV linear_native(weight, A[m]) for every row m, across M spanning multiple row-tiles.
    // Weight blocks vary their f16 d per block so columns are distinguishable (catches col mixups).

    // Build one valid native block of `dtype` with f16 scale `d` and a varied payload from `seed`.
    fn native_block(dtype: infr_core::DType, d: f32, seed: u8) -> Vec<u8> {
        use infr_core::DType::*;
        let dbits = half::f16::from_f32(d).to_bits().to_le_bytes();
        match dtype {
            Q8_0 => {
                let mut b = vec![0u8; 34];
                b[0..2].copy_from_slice(&dbits);
                fill(&mut b[2..34], 17, seed);
                b
            }
            Q4K => {
                let mut b = vec![0u8; 144];
                b[0..2].copy_from_slice(&dbits); // d
                b[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes()); // dmin
                fill(&mut b[4..16], 13, seed); // 6-bit scales
                fill(&mut b[16..144], 7, seed); // qs
                b
            }
            Q6K => {
                let mut b = vec![0u8; 210];
                fill(&mut b[0..128], 7, seed); // ql
                fill(&mut b[128..192], 11, seed); // qh
                fill(&mut b[192..208], 3, seed); // i8 scales
                b[208..210].copy_from_slice(&dbits); // d
                b
            }
            other => panic!("native_block: add {other:?}"),
        }
    }

    fn check_native_gemm(dtype: infr_core::DType, m: usize) {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        use infr_vulkan::linear::pad_to_u32_align;
        let n = 64usize;
        let k = 256usize;
        let belems = if dtype == infr_core::DType::Q8_0 {
            32
        } else {
            256
        };
        let blocks_per_row = k / belems;

        // Weight [N, K] as native blocks (row-major). d varies per block → distinguishable columns.
        let mut wbytes: Vec<u8> = Vec::new();
        for o in 0..n {
            for bk in 0..blocks_per_row {
                let d = 0.005 * ((o % 7) as f32 + 1.0) + 0.001 * bk as f32;
                wbytes.extend_from_slice(&native_block(dtype, d, (o * 3 + bk * 5) as u8));
            }
        }
        let wbuf = be.upload_weight_bytes(&pad_to_u32_align(&wbytes)).unwrap();

        // Activations [M, K], varied per (row, col).
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05 + ((i / k) as f32) * 0.001)
            .collect();
        let abuf = be.alloc(a.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(abuf.as_ref(), bytemuck::cast_slice(&a)).unwrap();

        // GPU GEMM → C [ceil(m/64)*64, N]. Device-local (coopmat store needs it), download via copy.
        let crows = m.div_ceil(64) * 64;
        let cbuf = be.alloc(crows * n * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.matmul_native(dtype, abuf.as_ref(), wbuf.as_ref(), cbuf.as_ref(), m, k, n);
        rec.finish().unwrap();
        let mut cbytes = vec![0u8; crows * n * 4];
        be.download(cbuf.as_ref(), &mut cbytes).unwrap();
        let cgemm: &[f32] = bytemuck::cast_slice(&cbytes);

        // Reference: one GEMV per row → C[m,:]
        for row in 0..m {
            let xbuf = be.alloc(k * 4, BufferUsage::Staging).unwrap();
            be.upload(
                xbuf.as_ref(),
                bytemuck::cast_slice(&a[row * k..row * k + k]),
            )
            .unwrap();
            let ybuf = be.alloc(n * 4, BufferUsage::Readback).unwrap();
            let rec2 = be.recorder().unwrap();
            rec2.linear_native(dtype, wbuf.as_ref(), xbuf.as_ref(), ybuf.as_ref(), 1, k, n);
            rec2.finish().unwrap();
            let mut ybytes = vec![0u8; n * 4];
            be.download(ybuf.as_ref(), &mut ybytes).unwrap();
            let yref: &[f32] = bytemuck::cast_slice(&ybytes);
            // The GEMM rounds activations+weights to f16 for coopmat (GEMV keeps f32 activations), so
            // compare error against the row's largest magnitude (standard GEMM metric) — near-zero
            // outputs from cancellation otherwise blow up a pure relative error.
            let rmax = yref.iter().fold(0f32, |a, &v| a.max(v.abs()));
            for col in 0..n {
                let g = cgemm[row * n + col];
                let r = yref[col];
                let err = (g - r).abs();
                assert!(
                    err < 0.02 * rmax + 1e-4,
                    "{dtype:?} GEMM vs GEMV at [{row},{col}]: gemm={g} gemv={r} err={err} rmax={rmax}"
                );
            }
        }
    }

    #[test]
    fn q8_0_native_gemm_matches_gemv() {
        check_native_gemm(infr_core::DType::Q8_0, 70);
    }

    #[test]
    fn q4k_native_gemm_matches_gemv() {
        check_native_gemm(infr_core::DType::Q4K, 70);
    }

    #[test]
    fn q6k_native_gemm_matches_gemv() {
        check_native_gemm(infr_core::DType::Q6K, 70);
    }

    // ── Native-block codebook formats (IQ4_NL/XS, MXFP4, NVFP4, TQ1_0, TQ2_0) ────
    //
    // CPU reference is `dequant_codebook` (the verified host port). GPU runs `linear_native`
    // with x=all-1.0 so the output is the sum of dequantized weights.

    fn check_native_cb(dtype: infr_core::DType, block_bytes: &[u8]) {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        use infr_vulkan::linear::pad_to_u32_align;
        let cpu = dequant_codebook(dtype, block_bytes);
        let numel = cpu.len();
        let cpu_out: f32 = cpu.iter().sum();

        let padded = pad_to_u32_align(block_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let x: Vec<f32> = vec![1.0f32; numel];
        let xbuf = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            dtype,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            numel,
            1,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_out: f32 = bytemuck::cast_slice(&out_bytes)[0];
        let rel = (gpu_out - cpu_out).abs() / (cpu_out.abs() + 1e-4);
        assert!(
            rel < 5e-3,
            "{dtype:?} native cb GPU vs CPU: gpu={gpu_out} cpu={cpu_out} rel={rel}"
        );
    }

    // varied non-trivial byte pattern
    fn fill(buf: &mut [u8], mul: u8, add: u8) {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(mul).wrapping_add(add);
        }
    }

    #[test]
    fn iq4nl_native_matches_cpu() {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.5).to_bits().to_le_bytes());
        fill(&mut block[2..18], 23, 7);
        check_native_cb(infr_core::DType::Iq4Nl, &block);
    }

    #[test]
    fn iq4xs_native_matches_cpu() {
        let mut block = vec![0u8; 136];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&0x9ce3u16.to_le_bytes()); // scales_h varied
        fill(&mut block[4..8], 53, 11); // scales_l
        fill(&mut block[8..136], 13, 3); // qs
        check_native_cb(infr_core::DType::Iq4Xs, &block);
    }

    #[test]
    fn mxfp4_native_matches_cpu() {
        let mut block = vec![0u8; 17];
        block[0] = 128; // e8m0 → d=1.0
        fill(&mut block[1..17], 29, 5);
        check_native_cb(infr_core::DType::Mxfp4, &block);
    }

    #[test]
    fn nvfp4_native_matches_cpu() {
        let mut block = vec![0u8; 36];
        block[0..4].copy_from_slice(&[0x38, 0x40, 0x48, 0x30]); // valid ue4m3 scales
        fill(&mut block[4..36], 19, 9);
        check_native_cb(infr_core::DType::Nvfp4, &block);
    }

    #[test]
    fn tq1_0_native_matches_cpu() {
        let mut block = vec![0u8; 54];
        fill(&mut block[0..52], 17, 1); // qs + qh
        block[52..54].copy_from_slice(&half::f16::from_f32(0.75).to_bits().to_le_bytes());
        check_native_cb(infr_core::DType::Tq1_0, &block);
    }

    #[test]
    fn tq2_0_native_matches_cpu() {
        let mut block = vec![0u8; 66];
        fill(&mut block[0..64], 11, 3); // qs
        block[64..66].copy_from_slice(&half::f16::from_f32(1.25).to_bits().to_le_bytes());
        check_native_cb(infr_core::DType::Tq2_0, &block);
    }

    #[test]
    fn iq2xxs_native_matches_cpu() {
        // 2 blocks (in_f=512) to exercise cross-block + grid/sign decode.
        let mut blocks = vec![0u8; 2 * 66];
        for (bi, blk) in blocks.chunks_mut(66).enumerate() {
            blk[0..2].copy_from_slice(
                &half::f16::from_f32(1.0 + bi as f32 * 0.5)
                    .to_bits()
                    .to_le_bytes(),
            );
            fill(&mut blk[2..66], 31, (bi as u8) * 7 + 13); // qs (grid idx + signs + scale)
        }
        check_native_cb(infr_core::DType::Iq2Xxs, &blocks);
    }

    #[test]
    fn iq2xs_native_matches_cpu() {
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 29, 5); // qs (u16 grid idx + sign)
        fill(&mut block[66..74], 17, 1); // scales
        check_native_cb(infr_core::DType::Iq2Xs, &block);
    }

    #[test]
    fn iq2s_native_matches_cpu() {
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 23, 7); // qs (idx low) + sign bytes
        fill(&mut block[66..74], 13, 2); // qh
        fill(&mut block[74..82], 19, 1); // scales
        check_native_cb(infr_core::DType::Iq2S, &block);
    }

    #[test]
    fn iq3xxs_native_matches_cpu() {
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 7, 1); // qs (grid indices)
        fill(&mut block[66..98], 13, 3); // sas (scale+signs)
        check_native_cb(infr_core::DType::Iq3Xxs, &block);
    }

    #[test]
    fn iq3s_native_matches_cpu() {
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 11, 2); // qs
        fill(&mut block[66..74], 5, 1); // qh
        fill(&mut block[74..106], 17, 3); // signs
        fill(&mut block[106..110], 3, 1); // scales
        check_native_cb(infr_core::DType::Iq3S, &block);
    }

    #[test]
    fn iq1s_native_matches_cpu() {
        let mut block = vec![0u8; 50];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..34], 13, 1); // qs
        fill(&mut block[34..50], 23, 7); // qh (u16: grid hi bits + scale + delta)
        check_native_cb(infr_core::DType::Iq1S, &block);
    }

    #[test]
    fn iq1m_native_matches_cpu() {
        let mut block = vec![0u8; 56];
        fill(&mut block[0..32], 17, 3); // qs
        fill(&mut block[32..48], 11, 1); // qh
                                         // scales: nonzero so packed d != 0
        block[48..56].copy_from_slice(&[0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a, 0xf0, 0x3d]);
        check_native_cb(infr_core::DType::Iq1M, &block);
    }
}
