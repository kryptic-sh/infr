//! Qwen3.5 / Qwen3.6 (`qwen35`, aka Qwen3-Next): hybrid gated-DeltaNet linear-attention + gated
//! full-attention. See `docs/QWEN35.md`.
//!
//! Production path: [`SeamModel`] builds a backend-agnostic decode `Graph` (composite ops over typed
//! handles, see `infr_core::graph`) and runs it through whichever [`infr_core::backend::Backend`] is
//! loaded (CPU / Vulkan / Metal) — batched/chunked prefill + per-token decode, the same engine behind
//! `infr run` / `infr bench` / `infr serve`.

use crate::load_tensor_dequant;
use anyhow::{anyhow, bail, Context, Result};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::TensorDesc;
use infr_core::{DType, TensorId, WeightSource};
use infr_cpu::CpuBackend;
use infr_gguf::{Gguf, TensorBytes};
use infr_vulkan::VulkanBackend;

/// Parsed `qwen35` hyper-parameters (subset needed for the 0.8B dense model).
#[derive(Debug, Clone)]
pub struct Cfg {
    pub n_layer: usize,
    pub n_embd: usize,
    pub vocab: usize,
    pub eps: f32,
    // attention layers
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize, // key_length == value_length (256)
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rope_sections: [u32; 4],
    pub full_attn_interval: usize,
    // linear (gated DeltaNet) layers
    pub d_conv: usize,  // ssm conv kernel (4)
    pub d_state: usize, // head_k_dim (128)
    pub d_inner: usize, // value_dim (2048)
    pub n_group: usize, // num_k_heads (16)
    pub dt_rank: usize, // num_v_heads (16)
}

impl Cfg {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.metadata().str("general.architecture").unwrap_or("");
        if arch != "qwen35" {
            bail!("not a qwen35 model (arch={arch:?})");
        }
        let u = |k: &str| g.metadata().u64(&format!("qwen35.{k}"));
        let req = |k: &str| u(k).ok_or_else(|| anyhow!("missing qwen35.{k}"));
        let f = |k: &str| -> Option<f32> {
            g.metadata()
                .get(&format!("qwen35.{k}"))
                .and_then(|v| match v {
                    infr_core::MetaValue::F64(x) => Some(*x as f32),
                    infr_core::MetaValue::U64(x) => Some(*x as f32),
                    infr_core::MetaValue::I64(x) => Some(*x as f32),
                    _ => None,
                })
        };
        // rope.dimension_sections is an array [11,11,10,0]
        let sections: [u32; 4] = {
            let mut s = [0u32; 4];
            if let Some(arr) = g
                .metadata()
                .get("qwen35.rope.dimension_sections")
                .and_then(|v| v.as_arr())
            {
                for (i, v) in arr.iter().take(4).enumerate() {
                    s[i] = v.as_u64().unwrap_or(0) as u32;
                }
            }
            s
        };
        Ok(Cfg {
            n_layer: req("block_count")? as usize,
            n_embd: req("embedding_length")? as usize,
            vocab: 0, // filled from token_embd shape
            eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            n_head: req("attention.head_count")? as usize,
            n_kv: req("attention.head_count_kv")? as usize,
            head_dim: req("attention.key_length")? as usize,
            rope_dim: u("rope.dimension_count").unwrap_or(64) as usize,
            rope_theta: f("rope.freq_base").unwrap_or(1e7),
            rope_sections: sections,
            full_attn_interval: u("full_attention_interval").unwrap_or(4) as usize,
            d_conv: req("ssm.conv_kernel")? as usize,
            d_state: req("ssm.state_size")? as usize,
            d_inner: req("ssm.inner_size")? as usize,
            n_group: req("ssm.group_count")? as usize,
            dt_rank: req("ssm.time_step_rank")? as usize,
        })
    }

    /// Attention (vs linear/SSM) layer test: every `full_attn_interval`-th layer is full attention.
    pub fn is_attn_layer(&self, i: usize) -> bool {
        (i + 1).is_multiple_of(self.full_attn_interval)
    }
    pub fn num_k_heads(&self) -> usize {
        self.n_group
    }
    pub fn num_v_heads(&self) -> usize {
        self.dt_rank
    }
    pub fn head_k_dim(&self) -> usize {
        self.d_state
    }
    pub fn head_v_dim(&self) -> usize {
        self.d_inner / self.dt_rank
    }
    pub fn conv_channels(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }
}

// ─── qwen35 on the backend-agnostic seam (production) ─────────────────────────
//
// Builds a batched/chunked-prefill decode `Graph` (composite ops over typed handles) and runs it
// through whichever `Backend` is loaded (CPU / Vulkan / Metal). The gated-DeltaNet recurrence +
// depthwise conv state and the attention KV cache are model-owned `Input` buffers, mutated in place
// each step.

/// Per-layer weight + state handles into one decode graph (declared in upload order so each binds to
/// the matching uploaded buffer).
struct Q35LinH {
    attn_norm: TensorId,
    qkv: TensorId,
    gate: TensorId,
    conv1d: TensorId,
    alpha: TensorId,
    beta: TensorId,
    ssm_a: TensorId,
    dt_bias: TensorId,
    ssm_norm: TensorId,
    out: TensorId,
    post_norm: TensorId,
    ffn_gate: TensorId,
    ffn_up: TensorId,
    ffn_down: TensorId,
    n_ff: usize,
    conv_state: TensorId,
    s_state: TensorId,
}
struct Q35AttnH {
    attn_norm: TensorId,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_norm: TensorId,
    k_norm: TensorId,
    out: TensorId,
    post_norm: TensorId,
    ffn_gate: TensorId,
    ffn_up: TensorId,
    ffn_down: TensorId,
    n_ff: usize,
    k_cache: TensorId,
    v_cache: TensorId,
}
enum Q35LayerH {
    Lin(Q35LinH),
    Attn(Q35AttnH),
}

/// Render a plain user message through the qwen35 GGUF's own jinja chat template (falls back to
/// ChatML — qwen35's native format — if there's no template). So `infr run` / tests pass plain text.
pub fn render_chat(path: &std::path::Path, user: &str) -> Result<String> {
    render_chat_messages(path, &[("user", user)])
}

/// Render a multi-turn conversation `(role, content)` through the qwen35 GGUF's own jinja chat
/// template — the [`crate::model::ChatModel::render`] primitive for the qwen35 GPU + CPU paths, so
/// the shared [`crate::model::Chat`] can drive a history-based REPL. Errors if the GGUF has no usable
/// `tokenizer.chat_template`.
pub fn render_chat_messages(path: &std::path::Path, messages: &[(&str, &str)]) -> Result<String> {
    let g = Gguf::open(path).map_err(|e| anyhow!("open gguf: {e}"))?;
    let tok = crate::build_tokenizer(&g)?;
    let eos = g.metadata().u64("tokenizer.ggml.eos_token_id").unwrap_or(2) as u32;
    infr_chat::render_chat_jinja(&g, &tok, eos, messages, true).ok_or_else(|| {
        anyhow!(
            "model GGUF has no usable chat template (no `tokenizer.chat_template`, or it failed to \
             render — set INFR_DEBUG_CHAT=1 for details)."
        )
    })
}

/// Greedy pure-CPU generation for qwen35 / Qwen3-Next on the agnostic seam (no Vulkan). `prompt` is
/// the already-formatted text (see [`render_chat`]); returns timing/counts, text streams via `on_piece`.
/// Context rows a call needs: the whole prompt + the generation budget + the sampled tail.
fn plen_hint(prompt_ids: &[u32], n: usize) -> usize {
    prompt_ids.len() + n + 1
}

/// Turns a native-dtype GGUF tensor into a backend weight buffer (upload or zero-copy map).
pub type BindWeight<'a> = &'a dyn Fn(&dyn Backend, TensorBytes) -> Result<Box<dyn Buffer>>;

/// Load-once Qwen3-Next (qwen35) seam model: a backend + the model's native-dtype weight buffers +
/// tokenizer/config, constructed ONCE and reused across `generate` calls. This is the SINGLE
/// generation engine behind `infr run` (via [`crate::model::Qwen35Chat`] /
/// on any backend) AND `infr bench` — bench times exactly what run executes, so a
/// production-path change can never silently leave the bench measuring a dead path again. Only the
/// per-conversation state (conv history, DeltaNet S, KV cache, sized to prompt+n) is allocated per
/// call.
pub struct SeamModel {
    be: Box<dyn Backend>,
    g: Gguf,
    c: Cfg,
    token_embd: Vec<f32>,
    vocab: usize,
    tok: tokenizers::Tokenizer,
    eos: Option<u32>,
    im_end: Option<u32>,
    wbufs: Vec<Box<dyn Buffer>>,
    wspecs: Vec<(DType, usize)>,
    /// Persistent per-conversation state (conv history, DeltaNet S, KV cache) + the token ids FED
    /// through it. Unlike the dense KV cache, the recurrent state is an append-only summary — it
    /// can't rewind to a prefix — so a turn reuses it ONLY when the new prompt exactly EXTENDS the
    /// cached sequence; anything else zero-resets and prefills from scratch.
    state: Option<SeamState>,
}

/// See [`SeamModel::state`].
struct SeamState {
    conv_bufs: Vec<Option<Box<dyn Buffer>>>,
    s_bufs: Vec<Option<Box<dyn Buffer>>>,
    k_bufs: Vec<Option<Box<dyn Buffer>>>,
    v_bufs: Vec<Option<Box<dyn Buffer>>>,
    max_ctx: usize,
    /// Token ids fed through the recurrent state (prompt + generated-and-fed of the last turn).
    cached: Vec<u32>,
}

impl SeamModel {
    /// Open the GGUF and upload every weight tensor in its native dtype through `bind_weight`
    /// (Vulkan/Metal: alloc+upload; CPU: zero-copy mmap). The upload order MUST equal the `wpush`
    /// order in the graph build.
    pub fn load(
        be: Box<dyn Backend>,
        bind_weight: BindWeight,
        path: &std::path::Path,
    ) -> Result<Self> {
        let gg = Gguf::open(path).map_err(|e| anyhow!("open gguf: {e}"))?;
        let g = &gg;
        let c = Cfg::from_gguf(g)?;
        let (token_embd, te_shape) = load_tensor_dequant(g, "token_embd.weight")?;
        let vocab = te_shape[1];
        let tok = crate::build_tokenizer(g)?;
        let eos = g
            .metadata()
            .u64("tokenizer.ggml.eos_token_id")
            .map(|x| x as u32);
        let im_end = tok.token_to_id("<|im_end|>");
        let attn = |i: usize| c.is_attn_layer(i);
        // Weight-load progress: the backend's alloc(Weights) ticks it; open the scope around the
        // upload loop (no-op scope on backends without a display). Guard drops at end of `load`.
        let fp = crate::weights::weight_footprint(g);
        let _weight_pb = be.weight_progress(Some(fp.dense + fp.expert));
        // ── upload weights in native GGUF dtype (the backend dequants on read). Order MUST equal the
        //    `wpush` order in `build`. ──────────────────────────────────────────────────────────────────
        let mut wbufs: Vec<Box<dyn Buffer>> = Vec::new();
        let mut wspecs: Vec<(DType, usize)> = Vec::new();
        let mut wraw = |name: &str| -> Result<()> {
            let info = g
                .tensors()
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| anyhow!("tensor not found: {name}"))?
                .clone();
            let numel: usize = info.shape.iter().product();
            let tb = g.tensor_bytes_arc(name).map_err(|e| anyhow!("{e}"))?;
            wbufs.push(bind_weight(be.as_ref(), tb)?);
            wspecs.push((info.dtype, numel));
            Ok(())
        };
        for i in 0..c.n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            if attn(i) {
                for nm in [
                    "attn_norm.weight",
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_q_norm.weight",
                    "attn_k_norm.weight",
                    "attn_output.weight",
                    "post_attention_norm.weight",
                    "ffn_gate.weight",
                    "ffn_up.weight",
                    "ffn_down.weight",
                ] {
                    wraw(&p(nm))?;
                }
            } else {
                for nm in [
                    "attn_norm.weight",
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_conv1d.weight",
                    "ssm_alpha.weight",
                    "ssm_beta.weight",
                    "ssm_a",
                    "ssm_dt.bias",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                    "post_attention_norm.weight",
                    "ffn_gate.weight",
                    "ffn_up.weight",
                    "ffn_down.weight",
                ] {
                    wraw(&p(nm))?;
                }
            }
        }
        wraw("output_norm.weight")?;
        // lm_head: `output.weight`, or (tied) the quantized `token_embd.weight` mapped zero-copy and
        // dequantized per-row by `Op::Linear`.
        if g.tensors().iter().any(|t| t.name == "output.weight") {
            wraw("output.weight")?;
        } else {
            wraw("token_embd.weight")?;
        }

        Ok(SeamModel {
            be,
            g: gg,
            c,
            token_embd,
            vocab,
            tok,
            eos,
            im_end,
            wbufs,
            wspecs,
            state: None,
        })
    }

    /// Vulkan (the production GPU path): weights uploaded raw to VRAM, dequantized in-kernel.
    pub fn load_vulkan(path: &std::path::Path) -> Result<Self> {
        let be = VulkanBackend::new().map_err(|e| anyhow!("vulkan init: {e}"))?;
        Self::load(
            Box::new(be),
            &|be, tb| {
                let buf = be
                    .alloc(tb.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                be.upload(buf.as_ref(), &tb).map_err(|e| anyhow!("{e}"))?;
                Ok(buf)
            },
            path,
        )
    }

    /// CPU reference backend: weights mapped zero-copy from the GGUF mmap.
    pub fn load_cpu(path: &std::path::Path) -> Result<Self> {
        Self::load(
            Box::new(CpuBackend::new()),
            &|_, tb| Ok(CpuBackend::new().map_weight(tb)),
            path,
        )
    }

    /// Reference Metal backend (Apple GPU).
    #[cfg(target_os = "macos")]
    pub fn load_metal(path: &std::path::Path) -> Result<Self> {
        let be = infr_metal::MetalBackend::new().map_err(|e| anyhow!("metal init: {e}"))?;
        Self::load(
            Box::new(be),
            &|be, tb| {
                let buf = be
                    .alloc(tb.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                be.upload(buf.as_ref(), &tb).map_err(|e| anyhow!("{e}"))?;
                Ok(buf)
            },
            path,
        )
    }

    /// Generate a completion for the already-rendered `prompt` on the loaded model: batched/chunked
    /// prefill + per-token decode (greedy), streaming text to `on_piece`. Per-conversation state is
    /// allocated fresh (sized to prompt+n).
    pub fn generate(
        &mut self,
        prompt: &str,
        n: usize,
        on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        self.generate_constrained(prompt, n, None, on_piece)
    }

    /// [`generate`](Self::generate) with an optional llguidance grammar constraint (serve's forced
    /// tool_choice) applied to the decode.
    pub fn generate_constrained(
        &mut self,
        prompt: &str,
        n: usize,
        mut constraint: Option<&mut crate::grammar::Constraint>,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let be = self.be.as_ref();
        let g = &self.g;
        let c = &self.c;
        let token_embd = &self.token_embd;
        let vocab = self.vocab;
        let tok = &self.tok;
        let (eos, im_end) = (self.eos, self.im_end);
        let wspecs = &self.wspecs;
        let wbufs = &self.wbufs;
        let ne = c.n_embd;
        let cc = c.conv_channels();
        let di = c.d_inner;
        let (nk, kd) = (c.num_k_heads(), c.head_k_dim());
        let (nv, vd) = (c.num_v_heads(), c.head_v_dim());
        let key_dim = nk * kd;
        let (nh, nkv, hd) = (c.n_head, c.n_kv, c.head_dim);
        let eps = c.eps;
        let kk = c.d_conv;
        let enc = tok
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

        if std::env::var("INFR_Q35_TIMING").is_ok() {
            eprintln!("[q35dims] ne={ne} nv={nv} nk={nk} kd={kd} vd={vd} cc={cc} di={di} n_layer={} nh={nh} nkv={nkv} hd={hd}", c.n_layer);
        }
        let attn = |i: usize| c.is_attn_layer(i);
        let n_ff_of = |i: usize| -> Result<usize> {
            Ok(g.tensors()
                .iter()
                .find(|t| t.name == format!("blk.{i}.ffn_up.weight"))
                .context("ffn_up")?
                .shape[1])
        };

        // ── persistent per-conversation state (conv history, DeltaNet S, KV cache) ─────
        // The recurrence REQUIRES zeros at a fresh start (`be.alloc` is calloc-style, but a REUSED
        // session must re-zero explicitly on reset). Session reuse: the recurrent state is an
        // append-only summary of the fed tokens, so a turn continues from it ONLY when the new
        // prompt exactly EXTENDS the cached sequence (the multi-turn chat shape) — anything else
        // (divergent prompt, identical prompt, capacity overflow) resets to zeros + full prefill.
        let zeroed = |be: &dyn Backend, n: usize| -> Result<Box<dyn Buffer>> {
            let buf = be
                .alloc(n * 4, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(buf.as_ref(), bytemuck::cast_slice(&vec![0f32; n]))
                .map_err(|e| anyhow!("{e}"))?;
            Ok(buf)
        };
        // F16 variant for the KV caches (2 bytes/elem; f16 zero is all-zero bits).
        let zeroed16 = |be: &dyn Backend, n: usize| -> Result<Box<dyn Buffer>> {
            let buf = be
                .alloc(n * 2, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(buf.as_ref(), bytemuck::cast_slice(&vec![0u16; n]))
                .map_err(|e| anyhow!("{e}"))?;
            Ok(buf)
        };
        let want_ctx = plen_hint(&prompt_ids, n);
        let reusable = matches!(&self.state, Some(st) if st.max_ctx >= want_ctx);
        if !reusable {
            // (Re)allocate at want + headroom so a growing conversation doesn't realloc per turn.
            let cap = want_ctx + 4096;
            let mut conv_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
            let mut s_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
            let mut k_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
            let mut v_bufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
            for i in 0..c.n_layer {
                if attn(i) {
                    k_bufs.push(Some(zeroed16(be, cap * nkv * hd)?));
                    v_bufs.push(Some(zeroed16(be, cap * nkv * hd)?));
                    conv_bufs.push(None);
                    s_bufs.push(None);
                } else {
                    conv_bufs.push(Some(zeroed(be, (kk - 1) * cc)?));
                    s_bufs.push(Some(zeroed(be, nv * kd * vd)?));
                    k_bufs.push(None);
                    v_bufs.push(None);
                }
            }
            self.state = Some(SeamState {
                conv_bufs,
                s_bufs,
                k_bufs,
                v_bufs,
                max_ctx: cap,
                cached: Vec::new(),
            });
        }
        let st = self.state.as_mut().expect("seam state just ensured");
        // Reuse test: the new prompt must strictly extend the fed sequence.
        let pfx = st
            .cached
            .iter()
            .zip(&prompt_ids)
            .take_while(|(a, b)| a == b)
            .count();
        let start = if pfx == st.cached.len() && pfx < prompt_ids.len() {
            pfx
        } else {
            if !st.cached.is_empty() {
                // Divergent (or fully-identical) prompt: the recurrent state can't rewind — zero it.
                let zero_fill = |b: &Option<Box<dyn Buffer>>, n: usize| -> Result<()> {
                    if let Some(b) = b {
                        be.upload(b.as_ref(), bytemuck::cast_slice(&vec![0f32; n]))
                            .map_err(|e| anyhow!("{e}"))?;
                    }
                    Ok(())
                };
                // KV caches are F16 (2 bytes/elem).
                let zero_fill16 = |b: &Option<Box<dyn Buffer>>, n: usize| -> Result<()> {
                    if let Some(b) = b {
                        be.upload(b.as_ref(), bytemuck::cast_slice(&vec![0u16; n]))
                            .map_err(|e| anyhow!("{e}"))?;
                    }
                    Ok(())
                };
                for i in 0..c.n_layer {
                    if attn(i) {
                        zero_fill16(&st.k_bufs[i], st.max_ctx * nkv * hd)?;
                        zero_fill16(&st.v_bufs[i], st.max_ctx * nkv * hd)?;
                    } else {
                        zero_fill(&st.conv_bufs[i], (kk - 1) * cc)?;
                        zero_fill(&st.s_bufs[i], nv * kd * vd)?;
                    }
                }
                st.cached.clear();
            }
            0
        };
        let max_ctx = st.max_ctx;
        let (conv_bufs, s_bufs, k_bufs, v_bufs) =
            (&st.conv_bufs, &st.s_bufs, &st.k_bufs, &st.v_bufs);

        // ── per-step IO ───────────────────────────────────────────────────────────────
        // Prompt tokens are ingested in chunks of up to CHUNK through ONE graph build each (instead of
        // one build per token); the recurrent conv/DeltaNet ops scan the chunk internally and the KV
        // cache is written for the whole chunk. Decode then runs one token (rows=1) at a time. The
        // hidden/pos input buffers are allocated per call sized to the chunk (the CPU backend writes
        // F32 inputs back, so their length must match the graph input exactly).
        // 512 beats 128 by ~27% end-to-end: the recurrent scans (DeltaNet/conv1d) cost the same total
        // either way, but bigger chunks give the GEMMs more row-tiles (occupancy) and amortize the
        // per-chunk build+compile+submit. INFR_Q35_CHUNK overrides for experiments.
        let chunk: usize = std::env::var("INFR_Q35_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&c: &usize| c > 0)
            .unwrap_or(512);
        let logits_buf = be
            .alloc(vocab * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;

        let f32d = |x: usize| TensorDesc::new(vec![x], DType::F32);
        let scale = 1.0 / (hd as f32).sqrt();

        // Build the graph that ingests `rows` tokens starting at absolute position `pos`
        // (kv_len = pos+rows). rows=1 is one decode token; rows>1 is a prefill chunk. Only the LAST
        // row's logits are produced (the next-token prediction).
        let build = |pos: usize,
                     rows: usize|
         -> Result<(Graph, TensorId, TensorId, Vec<TensorId>, TensorId)> {
            let mut gr = Graph::new();
            let hidden = gr.input(f32d(rows * ne));
            let positions = gr.input(TensorDesc::new(vec![rows], DType::I32));
            // weights in upload order
            let mut weights: Vec<TensorId> = Vec::new();
            let mut wi = 0usize;
            let mut wpush = |gr: &mut Graph, weights: &mut Vec<TensorId>| -> TensorId {
                let (dt, num) = wspecs[wi];
                wi += 1;
                let id = gr.weight(TensorDesc::new(vec![num], dt));
                weights.push(id);
                id
            };
            let mut layers: Vec<Q35LayerH> = Vec::new();
            for i in 0..c.n_layer {
                if attn(i) {
                    let attn_norm = wpush(&mut gr, &mut weights);
                    let q = wpush(&mut gr, &mut weights);
                    let k = wpush(&mut gr, &mut weights);
                    let v = wpush(&mut gr, &mut weights);
                    let q_norm = wpush(&mut gr, &mut weights);
                    let k_norm = wpush(&mut gr, &mut weights);
                    let out = wpush(&mut gr, &mut weights);
                    let post_norm = wpush(&mut gr, &mut weights);
                    let ffn_gate = wpush(&mut gr, &mut weights);
                    let ffn_up = wpush(&mut gr, &mut weights);
                    let ffn_down = wpush(&mut gr, &mut weights);
                    // F16 KV (matches the dense seam and llama.cpp): halves cache bandwidth
                    // AND is what the fast attention kernels key on — the f32 cache routed
                    // every GQA layer to the scalar split (measured 33 ms/call at d4096 on
                    // Metal, 52% of the whole decode).
                    let k_cache = gr.input(TensorDesc::new(vec![max_ctx * nkv * hd], DType::F16));
                    let v_cache = gr.input(TensorDesc::new(vec![max_ctx * nkv * hd], DType::F16));
                    layers.push(Q35LayerH::Attn(Q35AttnH {
                        attn_norm,
                        q,
                        k,
                        v,
                        q_norm,
                        k_norm,
                        out,
                        post_norm,
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        n_ff: n_ff_of(i)?,
                        k_cache,
                        v_cache,
                    }));
                } else {
                    let attn_norm = wpush(&mut gr, &mut weights);
                    let qkv = wpush(&mut gr, &mut weights);
                    let gate = wpush(&mut gr, &mut weights);
                    let conv1d = wpush(&mut gr, &mut weights);
                    let alpha = wpush(&mut gr, &mut weights);
                    let beta = wpush(&mut gr, &mut weights);
                    let ssm_a = wpush(&mut gr, &mut weights);
                    let dt_bias = wpush(&mut gr, &mut weights);
                    let ssm_norm = wpush(&mut gr, &mut weights);
                    let out = wpush(&mut gr, &mut weights);
                    let post_norm = wpush(&mut gr, &mut weights);
                    let ffn_gate = wpush(&mut gr, &mut weights);
                    let ffn_up = wpush(&mut gr, &mut weights);
                    let ffn_down = wpush(&mut gr, &mut weights);
                    let conv_state = gr.input(f32d((kk - 1) * cc));
                    let s_state = gr.input(f32d(nv * kd * vd));
                    layers.push(Q35LayerH::Lin(Q35LinH {
                        attn_norm,
                        qkv,
                        gate,
                        conv1d,
                        alpha,
                        beta,
                        ssm_a,
                        dt_bias,
                        ssm_norm,
                        out,
                        post_norm,
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        n_ff: n_ff_of(i)?,
                        conv_state,
                        s_state,
                    }));
                }
            }
            let w_out_norm = wpush(&mut gr, &mut weights);
            let w_lm = wpush(&mut gr, &mut weights);
            let logits = gr.output(f32d(vocab));

            // scratch — every activation buffer holds the whole `rows`-token chunk.
            let xn = gr.internal(f32d(rows * ne));
            let hn = gr.internal(f32d(rows * ne));
            let sub = gr.internal(f32d(rows * ne));
            let max_ff = (0..c.n_layer)
                .map(|i| n_ff_of(i).unwrap_or(0))
                .max()
                .unwrap_or(0);
            let gbuf = gr.internal(f32d(rows * max_ff));
            let ubuf = gr.internal(f32d(rows * max_ff));
            let actbuf = gr.internal(f32d(rows * max_ff));
            // linear-mixer scratch
            let qkvbuf = gr.internal(f32d(rows * cc));
            let zbuf = gr.internal(f32d(rows * di));
            let convout = gr.internal(f32d(rows * cc));
            let qbuf = gr.internal(f32d(rows * key_dim));
            let kbuf = gr.internal(f32d(rows * key_dim));
            let vbuf = gr.internal(f32d(rows * nv * vd));
            let bbuf = gr.internal(f32d(rows * nv));
            let abuf = gr.internal(f32d(rows * nv));
            let dnout = gr.internal(f32d(rows * nv * vd));
            // attn scratch
            let qg = gr.internal(f32d(rows * nh * 2 * hd));
            let qa = gr.internal(f32d(rows * nh * hd));
            let gate_a = gr.internal(f32d(rows * nh * hd));
            let ka = gr.internal(f32d(rows * nkv * hd));
            // K-norm output MUST be a dedicated F16 scratch so the Vulkan `kv_write_peephole` fuses the
            // QkNormRope+WriteKv into a direct-to-cache write (the peephole only fires on an F16 Internal
            // dst). Reusing the F32 `ka` here would make QkNormRope write f16 into an f32 buffer with NO
            // fusion, so the following WriteKv's `store_f16` reads those f16 bytes AS f32 → garbage cache.
            let k16 = gr.internal(TensorDesc::new(vec![rows * nkv * hd], DType::F16));
            let va = gr.internal(f32d(rows * nkv * hd));
            let attno = gr.internal(f32d(rows * nh * hd));

            // rmsn/lin run over the whole `rows`-token chunk (RmsNorm.rows / Linear.m = rows).
            let rmsn = |gr: &mut Graph, x: TensorId, w: TensorId, dst: TensorId| {
                gr.push(Op::RmsNorm {
                    x,
                    weight: w,
                    dst,
                    rows: rows as u32,
                    dim: ne as u32,
                    eps,
                });
            };
            let lin = |gr: &mut Graph,
                       x: TensorId,
                       w: TensorId,
                       dst: TensorId,
                       inf: usize,
                       outf: usize| {
                gr.push(Op::Linear {
                    x,
                    weight: w,
                    dst,
                    m: rows as u32,
                    in_f: inf as u32,
                    out_f: outf as u32,
                    w_off: 0,
                });
            };

            let __maxl = std::env::var("INFR_Q35_MAXLAYERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(usize::MAX);
            for (li, lh) in layers.iter().enumerate() {
                if li >= __maxl {
                    break;
                }
                match lh {
                    Q35LayerH::Lin(w) => {
                        rmsn(&mut gr, hidden, w.attn_norm, xn);
                        lin(&mut gr, xn, w.qkv, qkvbuf, ne, cc);
                        lin(&mut gr, xn, w.gate, zbuf, ne, di);
                        gr.push(Op::Conv1dSilu {
                            x: qkvbuf,
                            weight: w.conv1d,
                            state: w.conv_state,
                            dst: convout,
                            rows: rows as u32,
                            channels: cc as u32,
                            kernel: kk as u32,
                        });
                        // split conv_out [rows, cc=q|k|v] → packed [rows, *] q / k / v (strided per token).
                        gr.push(Op::CopyStrided {
                            src: convout,
                            src_off: 0,
                            src_stride: cc as u32,
                            dst: qbuf,
                            dst_off: 0,
                            dst_stride: key_dim as u32,
                            rows: rows as u32,
                            n: key_dim as u32,
                        });
                        gr.push(Op::CopyStrided {
                            src: convout,
                            src_off: key_dim as u32,
                            src_stride: cc as u32,
                            dst: kbuf,
                            dst_off: 0,
                            dst_stride: key_dim as u32,
                            rows: rows as u32,
                            n: key_dim as u32,
                        });
                        gr.push(Op::CopyStrided {
                            src: convout,
                            src_off: 2 * key_dim as u32,
                            src_stride: cc as u32,
                            dst: vbuf,
                            dst_off: 0,
                            dst_stride: (nv * vd) as u32,
                            rows: rows as u32,
                            n: (nv * vd) as u32,
                        });
                        lin(&mut gr, xn, w.beta, bbuf, ne, nv);
                        lin(&mut gr, xn, w.alpha, abuf, ne, nv);
                        gr.push(Op::DeltaNet {
                            q: qbuf,
                            k: kbuf,
                            v: vbuf,
                            b: bbuf,
                            a: abuf,
                            a_coef: w.ssm_a,
                            dt_bias: w.dt_bias,
                            state: w.s_state,
                            dst: dnout,
                            rows: rows as u32,
                            n_vhead: nv as u32,
                            n_khead: nk as u32,
                            head_k: kd as u32,
                            head_v: vd as u32,
                            eps: 1e-6,
                        });
                        // silu-gated RMSNorm per v-head: rmsnorm(out, ssm_norm) then * silu(z)
                        gr.push(Op::QkNorm {
                            x: dnout,
                            weight: w.ssm_norm,
                            dst: dnout,
                            rows: rows as u32,
                            n_head: nv as u32,
                            head_dim: vd as u32,
                            eps,
                        });
                        gr.push(Op::GatedAct {
                            gate: zbuf,
                            up: dnout,
                            dst: dnout,
                            rows: rows as u32,
                            nff: (nv * vd) as u32,
                            act: Activation::Silu,
                            up_off: 0,
                        });
                        lin(&mut gr, dnout, w.out, sub, di, ne);
                        gr.push(Op::Add {
                            a: hidden,
                            b: sub,
                            dst: hidden,
                            n: (rows * ne) as u32,
                        });
                        // FFN
                        rmsn(&mut gr, hidden, w.post_norm, hn);
                        lin(&mut gr, hn, w.ffn_gate, gbuf, ne, w.n_ff);
                        lin(&mut gr, hn, w.ffn_up, ubuf, ne, w.n_ff);
                        gr.push(Op::GatedAct {
                            gate: gbuf,
                            up: ubuf,
                            dst: actbuf,
                            rows: rows as u32,
                            nff: w.n_ff as u32,
                            act: Activation::Silu,
                            up_off: 0,
                        });
                        lin(&mut gr, actbuf, w.ffn_down, sub, w.n_ff, ne);
                        gr.push(Op::Add {
                            a: hidden,
                            b: sub,
                            dst: hidden,
                            n: (rows * ne) as u32,
                        });
                    }
                    Q35LayerH::Attn(w) => {
                        rmsn(&mut gr, hidden, w.attn_norm, xn);
                        // q proj outputs q+gate interleaved per head [h: q(hd), gate(hd)].
                        lin(&mut gr, xn, w.q, qg, ne, nh * 2 * hd);
                        // split interleaved per-head [q(hd) | gate(hd)] → packed qa / gate_a (strided).
                        for h in 0..nh {
                            gr.push(Op::CopyStrided {
                                src: qg,
                                src_off: (h * 2 * hd) as u32,
                                src_stride: (nh * 2 * hd) as u32,
                                dst: qa,
                                dst_off: (h * hd) as u32,
                                dst_stride: (nh * hd) as u32,
                                rows: rows as u32,
                                n: hd as u32,
                            });
                            gr.push(Op::CopyStrided {
                                src: qg,
                                src_off: (h * 2 * hd + hd) as u32,
                                src_stride: (nh * 2 * hd) as u32,
                                dst: gate_a,
                                dst_off: (h * hd) as u32,
                                dst_stride: (nh * hd) as u32,
                                rows: rows as u32,
                                n: hd as u32,
                            });
                        }
                        lin(&mut gr, xn, w.k, ka, ne, nkv * hd);
                        lin(&mut gr, xn, w.v, va, ne, nkv * hd);
                        // per-head q/k norm + RoPE — fused (qwen35 always has q/k-norm).
                        gr.push(Op::QkNormRope {
                            x: qa,
                            weight: w.q_norm,
                            positions,
                            dst: qa,
                            rows: rows as u32,
                            n_head: nh as u32,
                            head_dim: hd as u32,
                            rope_dim: c.rope_dim as u32,
                            theta: c.rope_theta,
                            eps,
                            freq_factors: None,
                        });
                        gr.push(Op::QkNormRope {
                            x: ka,
                            weight: w.k_norm,
                            positions,
                            dst: k16, // F16 scratch → peephole fuses this + the WriteKv below into a
                            // single direct-to-cache qk-norm+rope (see the k16 decl).
                            rows: rows as u32,
                            n_head: nkv as u32,
                            head_dim: hd as u32,
                            rope_dim: c.rope_dim as u32,
                            theta: c.rope_theta,
                            eps,
                            freq_factors: None,
                        });
                        gr.push(Op::WriteKv {
                            src: k16,
                            cache: w.k_cache,
                            rows: rows as u32,
                            row_stride: (nkv * hd) as u32,
                            pos: pos as u32,
                        });
                        gr.push(Op::WriteKv {
                            src: va,
                            cache: w.v_cache,
                            rows: rows as u32,
                            row_stride: (nkv * hd) as u32,
                            pos: pos as u32,
                        });
                        gr.push(Op::Attention {
                            q: qa,
                            k_cache: w.k_cache,
                            v_cache: w.v_cache,
                            dst: attno,
                            rows: rows as u32,
                            kv_len: (pos + rows) as u32,
                            n_head: nh as u32,
                            n_kv: nkv as u32,
                            head_dim: hd as u32,
                            scale,
                            mask: AttnMask::Causal,
                            pos: pos as u32,
                        });
                        // per-head sigmoid output gate
                        gr.push(Op::GatedAct {
                            gate: gate_a,
                            up: attno,
                            dst: attno,
                            rows: rows as u32,
                            nff: (nh * hd) as u32,
                            act: Activation::Sigmoid,
                            up_off: 0,
                        });
                        lin(&mut gr, attno, w.out, sub, nh * hd, ne);
                        gr.push(Op::Add {
                            a: hidden,
                            b: sub,
                            dst: hidden,
                            n: (rows * ne) as u32,
                        });
                        // FFN
                        rmsn(&mut gr, hidden, w.post_norm, hn);
                        lin(&mut gr, hn, w.ffn_gate, gbuf, ne, w.n_ff);
                        lin(&mut gr, hn, w.ffn_up, ubuf, ne, w.n_ff);
                        gr.push(Op::GatedAct {
                            gate: gbuf,
                            up: ubuf,
                            dst: actbuf,
                            rows: rows as u32,
                            nff: w.n_ff as u32,
                            act: Activation::Silu,
                            up_off: 0,
                        });
                        lin(&mut gr, actbuf, w.ffn_down, sub, w.n_ff, ne);
                        gr.push(Op::Add {
                            a: hidden,
                            b: sub,
                            dst: hidden,
                            n: (rows * ne) as u32,
                        });
                        let _ = li;
                    }
                }
            }
            // Only the LAST token's logits are needed (the next-token prediction). Extract its hidden
            // row, then output-norm + lm_head at rows=1 (avoids a wasteful [rows, vocab] projection).
            let last = gr.internal(f32d(ne));
            gr.push(Op::Copy {
                src: hidden,
                src_off: ((rows - 1) * ne) as u32,
                dst: last,
                dst_off: 0,
                n: ne as u32,
            });
            gr.push(Op::RmsNorm {
                x: last,
                weight: w_out_norm,
                dst: hn,
                rows: 1,
                dim: ne as u32,
                eps,
            });
            gr.push(Op::Linear {
                x: hn,
                weight: w_lm,
                dst: logits,
                m: 1,
                in_f: ne as u32,
                out_f: vocab as u32,
                w_off: 0,
            });

            // collect state-input handles per layer for binding (interleaved by kind)
            let mut state_ids: Vec<TensorId> = Vec::new();
            for lh in &layers {
                match lh {
                    Q35LayerH::Lin(w) => {
                        state_ids.push(w.conv_state);
                        state_ids.push(w.s_state);
                    }
                    Q35LayerH::Attn(w) => {
                        state_ids.push(w.k_cache);
                        state_ids.push(w.v_cache);
                    }
                }
            }
            Ok((gr, hidden, positions, [weights, state_ids].concat(), logits))
        };

        // ── drive ───────────────────────────────────────────────────────────────────
        // Build + compile + bind + execute the graph for the `rows` tokens whose embeddings/positions
        // are `emb` (rows*ne) / `posv` (rows), at absolute start position `pos`. Binds fresh hidden/pos
        // input buffers (sized to the chunk), the shared logits buffer, weights, and recurrent state.
        let run_graph = |pos: usize, emb: &[f32], posv: &[i32]| -> Result<()> {
            let rows = posv.len();
            let hidden_buf = be
                .alloc(emb.len() * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(hidden_buf.as_ref(), bytemuck::cast_slice(emb))
                .map_err(|e| anyhow!("{e}"))?;
            let pos_buf = be
                .alloc(posv.len() * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(pos_buf.as_ref(), bytemuck::cast_slice(posv))
                .map_err(|e| anyhow!("{e}"))?;
            let _t_build = std::time::Instant::now();
            let (gr, h_hidden, h_pos, h_bind, h_logits) = build(pos, rows)?;
            let plan = be.compile(&gr).map_err(|e| anyhow!("{e}"))?;
            let _bc = _t_build.elapsed();
            let _t_exec = std::time::Instant::now();
            let mut b = Bindings::new();
            b.bind(h_hidden, hidden_buf.as_ref());
            b.bind(h_pos, pos_buf.as_ref());
            b.bind(h_logits, logits_buf.as_ref());
            let nw = wbufs.len();
            for (i, id) in h_bind.iter().take(nw).enumerate() {
                b.bind(*id, wbufs[i].as_ref());
            }
            // state handles in layer order: per layer (conv,s) for linear, (k,v) for attn.
            let mut si = nw;
            for i in 0..c.n_layer {
                if attn(i) {
                    b.bind(h_bind[si], k_bufs[i].as_ref().unwrap().as_ref());
                    b.bind(h_bind[si + 1], v_bufs[i].as_ref().unwrap().as_ref());
                } else {
                    b.bind(h_bind[si], conv_bufs[i].as_ref().unwrap().as_ref());
                    b.bind(h_bind[si + 1], s_bufs[i].as_ref().unwrap().as_ref());
                }
                si += 2;
            }
            let r = be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"));
            if std::env::var("INFR_Q35_TIMING").is_ok() {
                eprintln!(
                    "[q35timing] rows={rows} build+compile={:.2}ms execute={:.2}ms",
                    _bc.as_secs_f64() * 1e3,
                    _t_exec.elapsed().as_secs_f64() * 1e3
                );
            }
            r
        };

        let mut outs: Vec<u32> = Vec::new();
        let mut logits = vec![0f32; vocab];
        let mut printed = 0usize; // streaming detok cursor
        let plen = prompt_ids.len();

        // ── prefill: ingest the prompt in chunks of ≤CHUNK (one graph per chunk) ──
        let prompt_t0 = std::time::Instant::now();
        let mut cpos = start;
        while cpos < plen {
            let rows = (plen - cpos).min(chunk);
            let mut emb = vec![0f32; rows * ne];
            for r in 0..rows {
                let tid = prompt_ids[cpos + r] as usize;
                emb[r * ne..r * ne + ne].copy_from_slice(&token_embd[tid * ne..tid * ne + ne]);
            }
            let posv: Vec<i32> = (0..rows).map(|r| (cpos + r) as i32).collect();
            run_graph(cpos, &emb, &posv)?;
            cpos += rows;
        }
        let prompt_t = prompt_t0.elapsed();
        // The last chunk's last row predicts the first generated token.
        be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
            .map_err(|e| anyhow!("{e}"))?;
        if std::env::var("INFR_Q35_DUMP").is_ok() {
            let am = argmax(&logits);
            let s: f32 = logits.iter().map(|x| x.abs()).sum::<f32>() / logits.len() as f32;
            eprintln!(
                "[q35dump] argmax={am} logit[am]={:.4} mean|logit|={s:.4} first6={:?}",
                logits[am as usize],
                &logits[..6]
            );
        }

        // ── decode: one token at a time (rows=1), feeding the last prediction back ──
        // Sampling: greedy unless INFR_TEMP is set (tests stay deterministic; the CLI sets chat
        // defaults for run/serve).
        let sampler = crate::sampling::Sampler::from_env();
        let mut rng = crate::sampling::seed_rng();
        let decode_t0 = std::time::Instant::now();
        let eos_list: Vec<u32> = eos.into_iter().chain(im_end).collect();
        let mut pos = plen;
        let mut decode_n = 0usize;
        loop {
            // Grammar-forced span (serve's tool_choice "required"/named): the shared llguidance
            // step; a step can carry several deterministically-forced tokens — feed each through
            // the graph in order.
            if let Some(cst) = constraint.as_deref_mut() {
                let (step, done) = crate::grammar::constrained_step(cst, &mut logits, &eos_list)?;
                if step.is_empty() {
                    break;
                }
                for &t in &step {
                    crate::stream_token(tok, &mut outs, &mut printed, t, &mut on_piece);
                    decode_n += 1;
                }
                if done || outs.len() >= n {
                    break;
                }
                for &t in &step {
                    let emb = &token_embd[t as usize * ne..t as usize * ne + ne];
                    run_graph(pos, emb, &[pos as i32])?;
                    pos += 1;
                }
                be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
                    .map_err(|e| anyhow!("{e}"))?;
                continue;
            }
            let next = crate::sampling::sample_logits(&logits, sampler, &mut rng);
            // Stop on EOS / <|im_end|> before emitting the stop token (chat turn boundary).
            // INFR_Q35_IGNORE_EOS keeps generating to the cap (benchmarks need a fixed tg count).
            if (Some(next) == eos || (im_end.is_some() && Some(next) == im_end))
                && std::env::var("INFR_Q35_IGNORE_EOS").is_err()
            {
                break;
            }
            crate::stream_token(tok, &mut outs, &mut printed, next, &mut on_piece);
            decode_n += 1;
            if outs.len() >= n {
                break;
            }
            // feed `next` at absolute position `pos`, predict the following token.
            let emb = &token_embd[next as usize * ne..next as usize * ne + ne];
            run_graph(pos, emb, &[pos as i32])?;
            be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
                .map_err(|e| anyhow!("{e}"))?;
            pos += 1;
        }
        let decode_t = decode_t0.elapsed();

        // Record the tokens FED through the recurrent state for the next turn's extend test:
        // the prompt plus every generated token that was fed back (the final sampled token — and
        // an EOS — never is).
        if let Some(st) = &mut self.state {
            st.cached = prompt_ids;
            if !outs.is_empty() {
                st.cached.extend_from_slice(&outs[..outs.len() - 1]);
            }
        }
        // The text streamed out via `on_piece`; return only timing/counts.
        Ok(crate::GenStats {
            // Tokens actually PREFILLED this call (the un-cached suffix) — the TTFT-honest count.
            n_prompt: plen - start,
            prompt_secs: prompt_t.as_secs_f64(),
            n_gen: decode_n,
            decode_secs: decode_t.as_secs_f64(),
        })
    }
}

/// Qwen3-Next decode on the CPU reference backend (weights mapped zero-copy from the GGUF mmap).
pub fn generate_cpu(
    path: &std::path::Path,
    prompt: &str,
    n: usize,
    on_piece: impl FnMut(&str),
) -> Result<crate::GenStats> {
    let mut m = SeamModel::load_cpu(path)?;
    m.generate(prompt, n, on_piece)
}

/// Qwen3-Next decode on the reference Metal backend (weights uploaded to Metal buffers in their
/// native GGUF dtype; the backend dequantizes lazily). The Apple-GPU twin of [`generate_cpu`].
#[cfg(target_os = "macos")]
pub fn generate_metal(
    path: &std::path::Path,
    prompt: &str,
    n: usize,
    on_piece: impl FnMut(&str),
) -> Result<crate::GenStats> {
    let mut m = SeamModel::load_metal(path)?;
    m.generate(prompt, n, on_piece)
}

/// Qwen3-Next on the Vulkan GPU through the AGNOSTIC SEAM (weights uploaded to VRAM in their native
/// GGUF dtype; the backend dequantizes in-kernel): batched/chunked prefill (the whole prompt flows
/// through a few graph builds instead of one per token) + per-token decode. The Vulkan twin of
/// [`generate_cpu`], and the production path behind [`generate_chat`]. See `gpu_seam_qwen35` and the
/// qwen35-batched-prefill memory.
pub fn generate_vulkan(
    path: &std::path::Path,
    prompt: &str,
    n: usize,
    on_piece: impl FnMut(&str),
) -> Result<crate::GenStats> {
    let mut m = SeamModel::load_vulkan(path)?;
    m.generate(prompt, n, on_piece)
}

/// True if the GGUF at `path` is a `qwen35` (Qwen3-Next) model.
pub fn is_qwen35(path: &std::path::Path) -> bool {
    Gguf::open(path)
        .ok()
        .map(|g| g.metadata().str("general.architecture") == Some("qwen35"))
        .unwrap_or(false)
}

fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::MIN;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locate the Qwen3.5-0.8B GGUF in the HF Hub cache (or `INFR_TEST_MODEL`), or `None` if it isn't
    /// present (the test self-skips).
    fn model_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
            return Some(std::path::PathBuf::from(p));
        }
        let hub = std::env::var("HOME").ok()? + "/.cache/huggingface/hub";
        let base = format!("{hub}/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots");
        std::fs::read_dir(&base).ok()?.find_map(|e| {
            let f = e.ok()?.path().join("Qwen3.5-0.8B-Q4_K_M.gguf");
            f.exists().then_some(f)
        })
    }

    /// Serialize the qwen35 tests: several toggle process-global env vars (e.g.
    /// `INFR_Q35_IGNORE_EOS`) mid-generate, which would otherwise race with a concurrently-running
    /// test in the same process. Poison-tolerant.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn loads_and_dims() {
        let _s = serial();
        let Some(path) = model_path() else {
            eprintln!("skip: Qwen3.5-0.8B not present");
            return;
        };
        let g = Gguf::open(&path).unwrap();
        let c = Cfg::from_gguf(&g).unwrap();
        println!("cfg: {c:?}");
        println!(
            "k_heads={} head_k={} v_heads={} head_v={} conv_ch={}",
            c.num_k_heads(),
            c.head_k_dim(),
            c.num_v_heads(),
            c.head_v_dim(),
            c.conv_channels()
        );
        assert_eq!(c.n_layer, 24);
        assert_eq!(c.conv_channels(), 6144);
        assert_eq!(c.head_v_dim(), 128);
        let n_attn = (0..c.n_layer).filter(|&i| c.is_attn_layer(i)).count();
        assert_eq!(n_attn, 6, "expected 6 full-attention layers");
    }

    /// Greedy generation on the seam's pure-CPU runner (`generate_cpu`, via `CpuBackend` — no GPU).
    #[test]
    fn greedy_generate() {
        let _s = serial();
        let Some(path) = model_path() else {
            eprintln!("skip: Qwen3.5-0.8B not present");
            return;
        };
        let prompt =
            std::env::var("Q35_PROMPT").unwrap_or_else(|_| "The capital of France is".to_string());
        let n = std::env::var("Q35_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);
        let mut out = String::new();
        generate_cpu(&path, &prompt, n, |p| out.push_str(p)).unwrap();
        println!("=== qwen35 CPU greedy ===\n{out}");
    }

    /// Bisection: run the seam on CPU (correct) then Vulkan with INFR_Q35_DUMP=1 and a chosen
    /// INFR_Q35_MAXLAYERS=N — compare the two `[q35dump]` argmax/logit lines to find the first layer
    /// count at which Vulkan diverges from CPU (localizes the wiring bug).
    #[test]
    #[ignore = "manual bisection (set INFR_Q35_DUMP + INFR_Q35_MAXLAYERS)"]
    fn q35_bisect() {
        let _s = serial();
        let Some(path) = model_path() else {
            return;
        };
        if !crate::gpu_available() {
            return;
        }
        let prompt = render_chat(&path, "Hi").unwrap();
        eprintln!("=== CPU seam ===");
        let _ = generate_cpu(&path, &prompt, 1, |_| {});
        eprintln!("=== Vulkan seam ===");
        let _ = generate_vulkan(&path, &prompt, 1, |_| {});
    }

    /// Session reuse on the qwen35 seam: turn 2's prompt EXTENDS turn 1's fed sequence, so the
    /// recurrent state continues (suffix-only prefill) and must generate exactly what a fresh
    /// full prefill of the same prompt generates. A divergent prompt resets (also exercised).
    #[test]
    #[ignore = "requires the Qwen3.5-0.8B GGUF + a Vulkan GPU"]
    fn gpu_seam_session_reuse() {
        let _s = serial();
        let Some(path) = model_path() else {
            eprintln!("skip: Qwen3.5-0.8B not present");
            return;
        };
        if !crate::gpu_available() {
            eprintln!("skip: no Vulkan GPU");
            return;
        }
        std::env::set_var("INFR_Q35_IGNORE_EOS", "1"); // fixed-length turns (no early stop)
        let mut m = SeamModel::load_vulkan(&path).unwrap();

        let p1 = "The quick brown fox jumps over the lazy dog. The capital of France is";
        let mut t1 = String::new();
        let s1 = m.generate(p1, 8, |p| t1.push_str(p)).unwrap();
        assert!(s1.n_prompt > 0);

        let p2 = format!("{p1}{t1} And the capital of Germany is");
        let mut t2 = String::new();
        let s2 = m.generate(&p2, 8, |p| t2.push_str(p)).unwrap();

        // fresh model = full prefill oracle
        let mut fresh = SeamModel::load_vulkan(&path).unwrap();
        let mut tf = String::new();
        fresh.generate(&p2, 8, |p| tf.push_str(p)).unwrap();
        assert_eq!(
            t2.trim(),
            tf.trim(),
            "session (suffix prefill on reused recurrent state) diverged from a fresh full prefill"
        );
        // suffix-only prefill: far fewer tokens than the whole prompt
        assert!(
            s2.n_prompt < s1.n_prompt,
            "turn 2 prefilled {} tokens — session reuse didn't kick in",
            s2.n_prompt
        );

        // divergent prompt → reset + full prefill (correctness, not reuse)
        let p3 = "Completely different subject: photosynthesis converts";
        let mut t3 = String::new();
        let s3 = m.generate(p3, 8, |p| t3.push_str(p)).unwrap();
        let mut fresh3 = SeamModel::load_vulkan(&path).unwrap();
        let mut tf3 = String::new();
        fresh3.generate(p3, 8, |p| tf3.push_str(p)).unwrap();
        assert_eq!(t3.trim(), tf3.trim(), "post-reset generation diverged");
        assert!(s3.n_prompt > 0);
        std::env::remove_var("INFR_Q35_IGNORE_EOS");
    }

    /// The batched/chunked-prefill SEAM on Vulkan: FAST (~450 pp / ~250 tg t/s) and correct — the
    /// K-norm output writes an F16 scratch so the Vulkan kv_write_peephole fuses QkNormRope+WriteKv
    /// (an F32 K-norm dst broke fusion → store_f16 read the f16 bytes as f32 → garbage K cache). This
    /// is the production GPU path (see `generate_chat`). See the qwen35-batched-prefill memory.
    #[test]
    #[ignore = "requires the Qwen3.5-0.8B GGUF + a Vulkan GPU"]
    fn gpu_seam_qwen35() {
        let Some(path) = model_path() else {
            eprintln!("skip: Qwen3.5-0.8B not present");
            return;
        };
        if !crate::gpu_available() {
            eprintln!("skip: no Vulkan GPU");
            return;
        }
        // Larger-prompt bench sample (~hundreds of tokens) so prefill t/s isn't dominated by fixed
        // per-chunk overhead. INFR_Q35_BENCH=1 blows the prompt up + skips the coherence assert.
        let (msg, ntok): (String, usize) = if std::env::var("INFR_Q35_BENCH").is_ok() {
            let filler = "The quick brown fox jumps over the lazy dog. ".repeat(80);
            (format!("Summarize this text in one word:\n{filler}"), 128)
        } else {
            ("What is bash? Answer briefly.".into(), 48)
        };
        let prompt = render_chat(&path, &msg).unwrap();
        let mut out = String::new();
        let stats = generate_vulkan(&path, &prompt, ntok, |p| out.push_str(p)).unwrap();
        eprintln!(
            "seam-vulkan: prefill {} tok @ {:.0} t/s | decode {} tok @ {:.1} t/s",
            stats.n_prompt,
            stats.n_prompt as f64 / stats.prompt_secs.max(1e-9),
            stats.n_gen,
            stats.n_gen as f64 / stats.decode_secs.max(1e-9),
        );
        eprintln!("seam-vulkan out: {out:?}");
        if std::env::var("INFR_Q35_BENCH").is_ok() {
            return;
        }
        let distinct = out
            .trim()
            .chars()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(distinct > 5, "seam-vulkan output degenerate: {out:?}");
    }
}
