//! Backend-agnostic compute IR.
//!
//! The model layer builds a [`Graph`] — an explicit, ordered list of semantic [`Op`]s over
//! typed [`TensorId`] handles — and a [`crate::backend::Backend`] compiles + executes it
//! however it likes (Vulkan SPIR-V, CPU loops, CUDA, ROCm, Metal, MLX). See PLAN.md
//! "The backend abstraction".
//!
//! ## Why an op-list, not a pure DAG
//!
//! The real transformer forward is imperative: it reuses scratch buffers, RoPEs in place, and
//! writes K/V into a persistent cache at a running offset. A pure SSA DAG can't express those
//! aliasing/stateful writes cleanly, so [`Graph`] is an **ordered list** of ops, each naming the
//! tensor handles it reads and the handle it writes (`dst`). Two ops may legally write the same
//! handle (in-place / scratch reuse) — order is significant, exactly like a command buffer.
//!
//! ## Composite ops
//!
//! Ops are *composite/semantic* (e.g. [`Op::Attention`], [`Op::QkNorm`]) rather than scalar
//! primitives, so a GPU backend can map each one straight to a hand-fused kernel (no perf loss)
//! while a CPU backend runs a plain loop. A future backend may either implement the composites
//! directly or add a lowering pass that decomposes them into primitives.

use crate::tensor::{TensorDesc, TensorId};

/// Attention masking mode. SWA layers (Gemma) mask beyond a sliding window; the rest are causal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnMask {
    /// Causal full attention (every position attends to all earlier positions).
    Causal,
    /// Causal sliding-window attention with the given window size (in tokens).
    SlidingWindow(usize),
    /// DiffusionGemma canvas denoise mask (bidirectional, NOT causal — see
    /// `docs/DIFFUSIONGEMMA.md`'s "Seam extensions" and the reference
    /// `llm_graph_input_attn_diffusion_decode::set_input` in `diffusion-gemma.cpp`): EVERY query
    /// row attends the SAME fixed range `[lo, kv_len)` regardless of its own row index — `pos`/
    /// per-row causal bounds are ignored entirely. `lo = 0` on full-attention layers (every
    /// prompt + canvas key visible); `lo = max(0, P - (n_swa-1))` on SWA layers (only the last
    /// `n_swa-1` prompt positions, but EVERY canvas key: canvas keys live in `[P, kv_len)` ⊆
    /// `[lo, kv_len)` on both layer types, since `lo <= P`). The caller (graph builder) computes
    /// `lo` from the prompt length `P` and the layer's SWA window; this variant only carries the
    /// already-resolved value.
    Canvas { lo: usize },
}

/// Activation used by the gated FFN (`act(gate) * up`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// SwiGLU: `silu(gate) * up` (Llama / Qwen).
    Silu,
    /// GeGLU: `gelu_tanh(gate) * up` (Gemma).
    Gelu,
    /// `sigmoid(gate) * up` (qwen35 output gate / silu-gated-RMSNorm uses Silu instead).
    Sigmoid,
}

/// How a tensor handle is provisioned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorKind {
    /// Per-step input bound at execute time via [`Bindings`] (e.g. the embedded hidden state,
    /// position ids, the KV cache). The backend does NOT allocate these.
    Input,
    /// Model weight bound from the loader via [`Bindings`]. Read-only.
    Weight,
    /// Backend-allocated scratch / activation, lives for the duration of one execute.
    Internal,
    /// An [`Internal`](TensorKind::Internal) tensor whose final value is read back by the caller
    /// (collected into [`Bindings::outputs`] after execute).
    Output,
}

/// Declaration of a tensor handle: its shape/dtype and how it's provisioned.
#[derive(Clone, Debug)]
pub struct TensorDecl {
    pub desc: TensorDesc,
    pub kind: TensorKind,
    /// Optional debug label (op/tensor name) for profiling + error messages.
    pub label: Option<String>,
}

/// Semantic ops. Each names the handles it reads plus the `dst` it writes. Grow as models need.
///
/// Dimensions that aren't derivable from the operand descs are carried inline (e.g. `n_head`,
/// `head_dim`) so a backend can execute an op without re-deriving layout from shapes.
#[derive(Clone, Debug)]
pub enum Op {
    /// `dst = rmsnorm(x) * weight`, normalizing over the last `dim` of each of `rows` rows.
    /// A weightless RMSNorm (Gemma V-norm) sets `weight` to a ones tensor.
    RmsNorm {
        x: TensorId,
        weight: TensorId,
        dst: TensorId,
        rows: u32,
        dim: u32,
        eps: f32,
    },
    /// `dst[m, out_f] = x[m, in_f] · weightᵀ`. `weight` may be any (quantized) dtype; the backend
    /// dispatches the kernel (GEMV/GEMM/MMQ on GPU, dequant+matvec on CPU).
    Linear {
        x: TensorId,
        weight: TensorId,
        dst: TensorId,
        m: u32,
        in_f: u32,
        out_f: u32,
        /// ELEMENT offset into `weight` where this projection's rows start (0 = whole tensor).
        /// Lets several projections share one concatenated weight upload (fused QKV): prefill
        /// runs ONE wide GEMM over the whole tensor while decode keeps per-projection GEMVs into
        /// its slices. Must be row-aligned (`w_off % in_f == 0`) and block-aligned for quants.
        w_off: u32,
    },
    /// Row-wise softmax: `dst[r, :] = softmax(x[r, :] * scale)` over `dim` columns, `rows` rows.
    /// diffusion-gemma's in-graph self-conditioning (see `docs/DIFFUSIONGEMMA.md`'s Phase-B and
    /// the reference's `dg_canvas_embed`): softmaxes the previous step's canvas logits over the
    /// FULL vocab before the soft-embedding matmul. `scale` is baked in so a temperature that
    /// changes per call doesn't need a separate `Scale` op ahead of this one — production code
    /// pre-multiplies on the host instead (keeps the compiled plan static across steps) and
    /// passes `scale: 1.0`, but the op itself is general.
    Softmax {
        x: TensorId,
        dst: TensorId,
        rows: u32,
        dim: u32,
        scale: f32,
    },
    /// Per-head RMSNorm of `x` (`rows × n_head × head_dim`) with a per-`head_dim` `weight`
    /// (Qwen3 / Gemma Q-norm and K-norm). In place when `dst == x`.
    QkNorm {
        x: TensorId,
        weight: TensorId,
        dst: TensorId,
        rows: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
    },
    /// NEOX RoPE over the first `rope_dim` of each head. `positions` is an i32 tensor of length
    /// `rows`. `freq_factors`, if present, divides per-pair angles (Gemma proportional RoPE).
    Rope {
        x: TensorId,
        positions: TensorId,
        dst: TensorId,
        rows: u32,
        n_head: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
        freq_factors: Option<TensorId>,
    },
    /// Fused per-head RMSNorm + NEOX RoPE — `QkNorm` immediately followed by `Rope` on the same
    /// tensor (the common qwen3/gemma q/k case). One pass: each head is rmsnormed (`× weight`) then
    /// its first `rope_dim` rotated, dims beyond `rope_dim` passing through normed. Maps 1:1 to the
    /// GPU's fused `qk_norm_rope` kernel; the CPU runs it as a single loop. Use the standalone
    /// `QkNorm` (gemma4 weightless V-norm, no RoPE) or `Rope` (llama, no q/k-norm) when not both.
    QkNormRope {
        x: TensorId,
        weight: TensorId,
        positions: TensorId,
        dst: TensorId,
        rows: u32,
        n_head: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
        eps: f32,
        freq_factors: Option<TensorId>,
    },
    /// Append `src` (`rows × row_stride`) into the persistent KV `cache` starting at row `pos`,
    /// casting to the cache dtype (typically f16). Stateful write — order matters.
    WriteKv {
        src: TensorId,
        cache: TensorId,
        rows: u32,
        row_stride: u32,
        pos: u32,
    },
    /// Scaled-dot-product attention. `q` is `rows × n_head × head_dim`; `k_cache`/`v_cache` hold
    /// `kv_len` rows of `n_kv × head_dim`. GQA when `n_head > n_kv`. `dst` is `rows × n_head ×
    /// head_dim`. `pos` is the absolute position of the first query row (for masking).
    Attention {
        q: TensorId,
        k_cache: TensorId,
        v_cache: TensorId,
        dst: TensorId,
        rows: u32,
        kv_len: u32,
        n_head: u32,
        n_kv: u32,
        head_dim: u32,
        scale: f32,
        mask: AttnMask,
        pos: u32,
    },
    /// Gated FFN activation: `dst[r,i] = act(gate[r,i]) * up[r, i + up_off]` (`rows × nff`). `gate`
    /// and `up` are separate handles (a backend may fuse them into one buffer internally). `up_off`
    /// shifts the `up` read by a whole-element offset so a layer-major slice of a bigger buffer can
    /// be consumed in place (Gemma E2B per-layer embedding); 0 for the normal case.
    GatedAct {
        gate: TensorId,
        up: TensorId,
        dst: TensorId,
        rows: u32,
        nff: u32,
        act: Activation,
        up_off: u32,
    },
    /// Gated FFN activation over a COMBINED `gu` buffer `[rows, 2*nff]` (gate half first, up half
    /// second per row): `dst[r,i] = act(gu[r,i]) * gu[r, nff+i]`. Produced when the runner
    /// concatenates the gate+up weights into one `[2*nff, ne]` tensor so the FFN input projection
    /// is a single GEMV/GEMM instead of two (see `Capabilities::combined_gu`).
    GatedActFused {
        gu: TensorId,
        dst: TensorId,
        rows: u32,
        nff: u32,
        act: Activation,
    },
    /// `dst[i] = a[i] + b[i]` (residual add). In place when `dst == a`.
    Add {
        a: TensorId,
        b: TensorId,
        dst: TensorId,
        n: u32,
    },
    /// Broadcast bias add: `dst[r*n + c] = x[r*n + c] + bias[c]` for `r` in `0..rows`, `c` in
    /// `0..n`. `bias` is a length-`n` vector added to every one of the `rows` rows (a projection's
    /// `Wx + b`). Qwen2/2.5 bias their q/k/v projections; the seam has no bias otherwise. In place
    /// when `dst == x`.
    AddBias {
        x: TensorId,
        bias: TensorId,
        dst: TensorId,
        rows: u32,
        n: u32,
    },
    /// `dst[i] = x[i] * s` (Gemma per-layer output scale, embedding scale).
    Scale {
        x: TensorId,
        dst: TensorId,
        s: f32,
        n: u32,
    },
    /// Broadcast elementwise multiply: `dst[r*n+c] = x[r*n+c] * vec[c]` for `r` in `0..rows`, `c`
    /// in `0..n` — the multiplicative twin of [`Op::AddBias`]. `vec` is a length-`n` weight
    /// (diffusion-gemma's router input scale `ffn_gate_inp.scale`, applied to the router's
    /// rmsnorm-noscale'd input before the router `Linear`; see `docs/DIFFUSIONGEMMA.md`).
    MulVec {
        x: TensorId,
        vec: TensorId,
        dst: TensorId,
        rows: u32,
        n: u32,
    },
    /// `dst[i] = cap * tanh(x[i] / cap)` (Gemma final-logit softcap).
    Softcap {
        x: TensorId,
        dst: TensorId,
        cap: f32,
        n: u32,
    },
    /// Copy `n` elements `src[src_off..] -> dst[dst_off..]` (extract last row, gather a slice).
    Copy {
        src: TensorId,
        src_off: u32,
        dst: TensorId,
        dst_off: u32,
        n: u32,
    },
    /// Batched strided copy: for `rows` rows, copy `n` elements
    /// `src[src_off + r*src_stride ..] -> dst[dst_off + r*dst_stride ..]`. Used to split a batched
    /// `[rows, cc]` interleaved buffer (e.g. conv output q|k|v) into packed `[rows, n]` slices in one
    /// op. `Copy` is the rows=1 special case.
    CopyStrided {
        src: TensorId,
        src_off: u32,
        src_stride: u32,
        dst: TensorId,
        dst_off: u32,
        dst_stride: u32,
        rows: u32,
        n: u32,
    },
    /// Mixture-of-experts FFN for a single token row (qwen3moe; diffusion-gemma's MoE branch — see
    /// `docs/DIFFUSIONGEMMA.md`). The router (`Linear` of `router_x[ne] → n_expert`) is softmaxed,
    /// the top-`n_used` experts selected, their softmax weights renormalized, and each runs a gated
    /// FFN on `x` (`act(gate·x) * (up·x)`, then `down·`); the outputs are summed weighted by the
    /// renormalized weights × `scale` into `dst[ne]` (the residual contribution).
    /// `gate_exps`/`up_exps`/`down_exps` are the stacked per-expert weights — expert `e` is the `e`-th
    /// equal byte slice (gate/up are `[n_ff_exp, ne]`, down is `[ne, n_ff_exp]` row-major).
    MoeFfn {
        x: TensorId,
        /// The router's own input row — usually the SAME tensor as `x` (qwen3moe: the router reads
        /// whatever normed input feeds the experts). diffusion-gemma's router reads a DIFFERENTLY
        /// normalized/scaled row of the same residual (`rmsnorm_noscale(attn_out)/√ne ·
        /// ffn_gate_inp.scale`, built with `Op::RmsNorm` + `Op::Scale` + `Op::MulVec` upstream), so
        /// it's a separate handle rather than reusing `x`.
        router_x: TensorId,
        router: TensorId,
        gate_exps: TensorId,
        /// Ignored when `fused_gate_up` is set (the call site passes the same handle as
        /// `gate_exps` — never read).
        up_exps: TensorId,
        down_exps: TensorId,
        /// Per-expert scale on the selected expert's DOWN-projection output BEFORE the weighted
        /// sum (diffusion-gemma `ffn_down_exps.scale[n_expert]`, one f32 per expert). `None` = no
        /// scale (qwen3moe).
        down_scale: Option<TensorId>,
        /// `gate_exps` holds gate AND up FUSED into one `[ne, 2*n_ff_exp, n_expert]` tensor (gate
        /// rows first, up rows second — the same "gate half, up half" per-expert-slice convention
        /// as `Op::GatedActFused`'s combined `gu` buffer). `up_exps` is unused when `true`
        /// (diffusion-gemma's `ffn_gate_up_exps`); `false` = separate `gate_exps`/`up_exps` tensors
        /// (qwen3moe).
        fused_gate_up: bool,
        dst: TensorId,
        ne: u32,
        n_expert: u32,
        n_used: u32,
        n_ff_exp: u32,
        scale: f32,
        act: Activation,
    },
    /// Depthwise causal 1-D conv over `channels` followed by SiLU (qwen35 gated DeltaNet).
    /// Processes `rows` tokens sequentially, carrying the rolling history in `state` across rows and
    /// leaving it updated after the last row. `x`/`dst` are `[rows, channels]`; `weight` is the
    /// per-channel kernel `[channels, kernel]`; `state` is the rolling `[(kernel-1), channels]`
    /// history (oldest row first). Per token: `dst[ch] = silu(Σ_{j<kernel-1} state[j,ch]·w[ch,j] +
    /// x[ch]·w[ch,K-1])`, then history shifts (drop oldest, append raw `x`). `rows=1` = one token.
    Conv1dSilu {
        x: TensorId,
        weight: TensorId,
        state: TensorId,
        dst: TensorId,
        rows: u32,
        channels: u32,
        kernel: u32,
    },
    /// Gated-DeltaNet linear-attention recurrence step (qwen35), one token. Per VALUE head:
    /// L2-normalize `q`,`k`; scale `q` by `1/√head_k`; `beta = sigmoid(b)`, `decay =
    /// exp(a_coef·softplus(a + dt_bias))`; update the persistent state `S[head_k, head_v]`: `S *=
    /// decay`, `delta = (v − Sᵀk)·beta`, `S += k⊗delta`; `dst = Sᵀq`. GQA linear attention: `n_vhead`
    /// value heads share `n_khead` query/key heads in contiguous groups of `n_vhead/n_khead` — value
    /// head `h` uses q/k head `h/(n_vhead/n_khead)`. `q`/`k` are `[n_khead·head_k]`, `v`/`dst` are
    /// `[n_vhead·head_v]`, `b`/`a` are `[n_vhead]`, `a_coef`/`dt_bias` are weights `[n_vhead]`,
    /// `state` is `[n_vhead·head_k·head_v]` (mutated in place). Processes `rows` tokens sequentially,
    /// carrying `state` across rows (and leaving it updated after the last). `q`/`k` are
    /// `[rows, n_khead·head_k]`, `v`/`dst` are `[rows, n_vhead·head_v]`, `b`/`a` are `[rows, n_vhead]`.
    /// `rows=1` = one token.
    DeltaNet {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        b: TensorId,
        a: TensorId,
        a_coef: TensorId,
        dt_bias: TensorId,
        state: TensorId,
        dst: TensorId,
        rows: u32,
        n_vhead: u32,
        n_khead: u32,
        head_k: u32,
        head_v: u32,
        eps: f32,
    },
}

impl Op {
    /// The op's variant name — used by backends for per-op profiling / error messages so the
    /// mapping lives in ONE place (was duplicated as `op_kind`/`op_name` in each backend).
    pub fn kind(&self) -> &'static str {
        match self {
            Op::RmsNorm { .. } => "RmsNorm",
            Op::Softmax { .. } => "Softmax",
            Op::Linear { .. } => "Linear",
            Op::QkNorm { .. } => "QkNorm",
            Op::Rope { .. } => "Rope",
            Op::QkNormRope { .. } => "QkNormRope",
            Op::WriteKv { .. } => "WriteKv",
            Op::Attention { .. } => "Attention",
            Op::GatedAct { .. } => "GatedAct",
            Op::GatedActFused { .. } => "GatedActFused",
            Op::Add { .. } => "Add",
            Op::AddBias { .. } => "AddBias",
            Op::Scale { .. } => "Scale",
            Op::MulVec { .. } => "MulVec",
            Op::Softcap { .. } => "Softcap",
            Op::Copy { .. } => "Copy",
            Op::CopyStrided { .. } => "CopyStrided",
            Op::MoeFfn { .. } => "MoeFfn",
            Op::Conv1dSilu { .. } => "Conv1dSilu",
            Op::DeltaNet { .. } => "DeltaNet",
        }
    }
}

/// An ordered op-list over declared tensor handles. Node index in `tensors` == [`TensorId`].
#[derive(Clone, Default)]
pub struct Graph {
    pub tensors: Vec<TensorDecl>,
    pub ops: Vec<Op>,
    pub inputs: Vec<TensorId>,
    pub weights: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// The `Input` tensors written IN PLACE by the graph's ops (the KV cache: `WriteKv`'s `cache` and
    /// `Attention`'s `k_cache`/`v_cache`). This is pure graph semantics, so it lives here rather than
    /// being rediscovered per backend: an eager-load backend (like the CPU interpreter) skips loading
    /// these into its working store + skips writing them back (they're mutated directly), avoiding
    /// O(max_ctx) copies per step.
    pub fn in_place_inputs(&self) -> std::collections::HashSet<TensorId> {
        let mut set = std::collections::HashSet::new();
        for op in &self.ops {
            match op {
                Op::WriteKv { cache, .. } => {
                    set.insert(*cache);
                }
                Op::Attention {
                    k_cache, v_cache, ..
                } => {
                    set.insert(*k_cache);
                    set.insert(*v_cache);
                }
                _ => {}
            }
        }
        set
    }

    fn decl(&mut self, desc: TensorDesc, kind: TensorKind) -> TensorId {
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(TensorDecl {
            desc,
            kind,
            label: None,
        });
        id
    }

    /// Declare a per-step input (bound at execute time).
    pub fn input(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.decl(desc, TensorKind::Input);
        self.inputs.push(id);
        id
    }

    /// Declare a model weight (bound from the loader).
    pub fn weight(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.decl(desc, TensorKind::Weight);
        self.weights.push(id);
        id
    }

    /// Declare backend-allocated scratch.
    pub fn internal(&mut self, desc: TensorDesc) -> TensorId {
        self.decl(desc, TensorKind::Internal)
    }

    /// Declare a read-back output.
    pub fn output(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.decl(desc, TensorKind::Output);
        self.outputs.push(id);
        id
    }

    /// Attach a debug label to a tensor handle.
    pub fn label(&mut self, id: TensorId, label: impl Into<String>) -> TensorId {
        self.tensors[id.0 as usize].label = Some(label.into());
        id
    }

    /// Append an op to the list.
    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
    }

    pub fn desc(&self, id: TensorId) -> &TensorDesc {
        &self.tensors[id.0 as usize].desc
    }

    pub fn kind(&self, id: TensorId) -> TensorKind {
        self.tensors[id.0 as usize].kind
    }
}
