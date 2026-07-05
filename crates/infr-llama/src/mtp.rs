//! MTP (multi-token prediction) head weights for qwen35 (issue #33 — see `docs/MTP.md`).
//!
//! Phase 1 scope: locate + shape-check the tensors the head's 1-layer graph will need. NO ops are
//! emitted here — Phase 2 builds the actual forward (its own 1-layer `Graph`, reusing the trunk's
//! full-attention emission verbatim) and will bind these tensor NAMES through the same `wload`/
//! `wpush` machinery `cpu_backend.rs`'s trunk loop already uses (see that module's `wload`
//! closure). This module only resolves WHICH names to bind (with the reference's fallback rule for
//! the three optional `nextn.*` tensors) and validates their shapes match what the reference
//! `qwen35.cpp::load_arch_tensors`'s `load_block_mtp` creates — so a shape drift in a future GGUF
//! export fails loudly here instead of silently misreading tensor bytes once Phase 2 builds ops
//! over them.
//!
//! The head layer sits at GGUF index `blk.{n_layer}` (`Config::n_layer` is already the TRUNK
//! count — see `Config::n_layer_nextn`'s doc), one index past the trunk's last layer, and carries
//! a FULL qwen35 attention-layer tensor set (same names/shapes/interleaved-q+gate layout as a
//! trunk full-attention layer, `docs/QWEN35.md`) plus the `nextn.*` bridging tensors.

use anyhow::{anyhow, bail, Result};
use infr_core::{TensorInfo, WeightSource};
use infr_gguf::Gguf;

/// One resolved-and-shape-checked tensor: its GGUF metadata (name/shape/dtype/offset), kept around
/// so Phase 2's `wload` can re-derive the bytes from `.name` without this module re-reading the
/// file or holding a second copy of the mmap slice.
pub type MtpTensor = TensorInfo;

/// The qwen35 MTP head's tensors (see the module doc). Every required field here EXISTED in the
/// GGUF and had the expected shape at [`load_mtp_head`] time; the three `Option` fields are the
/// ones the reference allows to fall back to the main model's tensors when absent (`docs/MTP.md`'s
/// confirmed dump: the shipped 4B GGUF omits `embed_tokens`/`shared_head_head` — those two fall
/// back — but DOES ship its own `shared_head_norm`).
pub struct MtpHeadWeights {
    /// The GGUF block index of the head layer (`cfg.n_layer`, i.e. immediately after the trunk).
    pub il: usize,
    // ── standard qwen35 full-attention layer tensors (identical shapes to a trunk full-attn
    //    layer at a `(il+1) % full_attn_interval == 0` index — see `Config::is_qwen35_attn_layer`) ──
    pub attn_norm: MtpTensor,
    /// Interleaved q+gate: `[n_embd, head_dim * n_head * 2]` (see `Config::attn_out_gate`).
    pub attn_q: MtpTensor,
    pub attn_k: MtpTensor,
    pub attn_v: MtpTensor,
    pub attn_q_norm: MtpTensor,
    pub attn_k_norm: MtpTensor,
    pub attn_output: MtpTensor,
    pub post_attention_norm: MtpTensor,
    pub ffn_gate: MtpTensor,
    pub ffn_up: MtpTensor,
    pub ffn_down: MtpTensor,
    // ── NextN bridge (the tensors that make this an MTP head, not just another trunk layer) ──
    /// `[2*n_embd, n_embd]`: projects `concat(rmsnorm(embed(t)), rmsnorm(h_target))` down to
    /// `n_embd` before the layer's own attention (see `docs/MTP.md`'s forward pseudocode).
    pub eh_proj: MtpTensor,
    pub enorm: MtpTensor,
    pub hnorm: MtpTensor,
    /// Falls back to the main model's `token_embd.weight` when absent (the shipped 4B GGUF has no
    /// `nextn.embed_tokens` — see `docs/MTP.md`'s confirmed dump).
    pub embed_tokens: Option<MtpTensor>,
    /// Falls back to the main model's (tied) lm_head when absent.
    pub shared_head_head: Option<MtpTensor>,
    /// Falls back to the main model's `output_norm.weight` when absent.
    pub shared_head_norm: Option<MtpTensor>,
}

fn find<'a>(g: &'a Gguf, name: &str) -> Option<&'a TensorInfo> {
    g.tensors().iter().find(|t| t.name == name)
}

fn require(g: &Gguf, name: &str, want: &[usize]) -> Result<MtpTensor> {
    let t = find(g, name).ok_or_else(|| anyhow!("MTP head: missing tensor {name}"))?;
    if t.shape != want {
        bail!(
            "MTP head: {name} has shape {:?}, expected {:?}",
            t.shape,
            want
        );
    }
    Ok(t.clone())
}

/// Like [`require`] but `None` (not an error) when the tensor is simply absent — the reference's
/// fallback tensors (`nextn.embed_tokens`/`shared_head_head`/`shared_head_norm`). A PRESENT tensor
/// with the wrong shape is still an error (a real corruption, not an intentional omission).
fn optional(g: &Gguf, name: &str, want: &[usize]) -> Result<Option<MtpTensor>> {
    match find(g, name) {
        Some(t) if t.shape == want => Ok(Some(t.clone())),
        Some(t) => bail!(
            "MTP head: {name} has shape {:?}, expected {:?}",
            t.shape,
            want
        ),
        None => Ok(None),
    }
}

/// Locate + shape-check the qwen35 MTP head's tensors (see the module doc). Requires
/// `cfg.n_layer_nextn == 1` (Phase 1's only supported case — `Config::from_gguf` already rejects
/// anything else) and `cfg.qwen35`.
pub fn load_mtp_head(g: &Gguf, cfg: &crate::Config) -> Result<MtpHeadWeights> {
    if !cfg.qwen35 || cfg.n_layer_nextn != 1 {
        bail!(
            "load_mtp_head: requires a qwen35 GGUF with nextn_predict_layers==1 (got qwen35={}, \
             n_layer_nextn={})",
            cfg.qwen35,
            cfg.n_layer_nextn,
        );
    }
    let il = cfg.n_layer; // the MTP head sits immediately after the trunk (see Config::n_layer_nextn)
    let p = |s: &str| format!("blk.{il}.{s}");
    let ne = cfg.n_embd;
    let qdim = cfg.head_dim * cfg.n_head * 2; // interleaved q+gate, see Config::attn_out_gate
    let kv_dim = cfg.n_kv * cfg.head_dim;
    Ok(MtpHeadWeights {
        il,
        attn_norm: require(g, &p("attn_norm.weight"), &[ne])?,
        attn_q: require(g, &p("attn_q.weight"), &[ne, qdim])?,
        attn_k: require(g, &p("attn_k.weight"), &[ne, kv_dim])?,
        attn_v: require(g, &p("attn_v.weight"), &[ne, kv_dim])?,
        attn_q_norm: require(g, &p("attn_q_norm.weight"), &[cfg.head_dim])?,
        attn_k_norm: require(g, &p("attn_k_norm.weight"), &[cfg.head_dim])?,
        attn_output: require(
            g,
            &p("attn_output.weight"),
            &[cfg.head_dim * cfg.n_head, ne],
        )?,
        post_attention_norm: require(g, &p("post_attention_norm.weight"), &[ne])?,
        ffn_gate: require(g, &p("ffn_gate.weight"), &[ne, cfg.n_ff])?,
        ffn_up: require(g, &p("ffn_up.weight"), &[ne, cfg.n_ff])?,
        ffn_down: require(g, &p("ffn_down.weight"), &[cfg.n_ff, ne])?,
        eh_proj: require(g, &p("nextn.eh_proj.weight"), &[2 * ne, ne])?,
        enorm: require(g, &p("nextn.enorm.weight"), &[ne])?,
        hnorm: require(g, &p("nextn.hnorm.weight"), &[ne])?,
        embed_tokens: optional(g, &p("nextn.embed_tokens.weight"), &[ne, cfg.vocab])?,
        shared_head_head: optional(g, &p("nextn.shared_head_head.weight"), &[ne, cfg.vocab])?,
        shared_head_norm: optional(g, &p("nextn.shared_head_norm.weight"), &[ne])?,
    })
}
