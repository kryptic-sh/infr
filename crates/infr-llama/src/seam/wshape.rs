//! The dims each loaded weight must have, derived from [`Config`] exactly as the decode graph
//! indexes it, so a GGUF whose tensor disagrees with its own metadata is refused at load instead
//! of being read out of bounds (or silently mis-strided) by the first forward pass.
//!
//! Every rule mirrors the `Op` that consumes the tensor in `runner.rs` — `Op::Linear`'s
//! `[in_f, out_f]`, an `Op::RmsNorm`'s `dim`, `Op::MoeFfn`'s expert banks — in GGUF order (`ne0`,
//! the contiguous axis, first). A tensor no rule here covers is NOT checked: those are the
//! tensors whose consumer has no single width to compare against, or whose layout the graph
//! re-interprets in a way this table would have to duplicate rather than derive (see
//! [`expected_weight_shape`]).
use crate::Config;
use anyhow::{anyhow, Result as AResult};

/// The shape `name` must have for `c`'s decode graph, or `None` when no rule covers it.
///
/// Deliberately unchecked: the DeepSeek MLA absorbed `attn_k_b`/`attn_v_b` (the `Op::Mla` kernels
/// read them in their own head-major layout), the V3.2 lightning indexer, every DeepSeek V4
/// mixer/compressor/hyper-connection tensor, the qwen35 DeltaNet tensors, the dual-FFN (gemma4
/// MoE / DiffusionGemma) router and expert banks and their extra norms, the gemma4 E2B per-layer
/// embedding tensors, DiffusionGemma's self-conditioning MLP, and a KV-sharing layer's own
/// `attn_k`/`attn_v` (loaded, but never read by the graph).
pub(super) fn expected_weight_shape(c: &Config, name: &str) -> Option<Vec<usize>> {
    let ne = c.n_embd;
    match name {
        "token_embd.weight" | "output.weight" => return Some(vec![ne, c.vocab]),
        "output_norm.weight" => return Some(vec![ne]),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (l, role) = rest.split_once('.')?;
    let l: usize = l.parse().ok()?;
    if l >= c.n_layer {
        return None;
    }
    let delta = c.qwen35 && !c.is_qwen35_attn_layer(l);
    let mla = c.deepseek2 && !c.deepseek4;
    // The standard attention mixer (`MixerW::Attn`): every arch but DeltaNet, MLA and V4.
    let attn = !delta && !c.deepseek2 && !c.deepseek4;
    let (hd, nkv, nff) = (c.layer_head_dim(l), c.layer_n_kv(l), c.layer_n_ff(l));
    let (qrow, kvrow) = (c.n_head * hd, nkv * hd);
    let moe_layer = !c.dual_moe() && c.moe.is_some() && c.is_moe_layer(l);
    let dense_ffn = !moe_layer;
    let shape = match role {
        "attn_norm.weight"
        | "ffn_norm.weight"
        | "post_attention_norm.weight"
        | "post_ffw_norm.weight" => vec![ne],
        // qwen35 packs a per-head sigmoid output gate beside each query head (`attn_out_gate`).
        "attn_q.weight" if attn => vec![ne, qrow * if c.attn_out_gate { 2 } else { 1 }],
        "attn_k.weight" | "attn_v.weight" if attn && c.has_own_kv(l) => vec![ne, kvrow],
        "attn_q.bias" if attn => vec![qrow],
        "attn_k.bias" | "attn_v.bias" if attn && c.has_own_kv(l) => vec![kvrow],
        "attn_q_norm.weight" | "attn_k_norm.weight" if attn => vec![hd],
        "attn_output.weight" if attn => vec![qrow, ne],
        "attn_sub_norm.weight" if attn => vec![qrow],
        "ffn_gate.weight" | "ffn_up.weight" if dense_ffn => vec![ne, nff],
        "ffn_down.weight" if dense_ffn => vec![nff, ne],
        "ffn_sub_norm.weight" if dense_ffn => vec![nff],
        "ffn_gate_inp.weight"
        | "ffn_gate_exps.weight"
        | "ffn_up_exps.weight"
        | "ffn_down_exps.weight"
        | "exp_probs_b.bias"
            if moe_layer =>
        {
            let mc = c.moe?;
            match role {
                "ffn_gate_inp.weight" => vec![ne, mc.n_expert],
                "ffn_down_exps.weight" => vec![mc.n_ff_exp, ne, mc.n_expert],
                "exp_probs_b.bias" => vec![mc.n_expert],
                _ => vec![ne, mc.n_ff_exp, mc.n_expert],
            }
        }
        "ffn_gate_inp_shexp.weight" if moe_layer => vec![ne],
        "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" if moe_layer => vec![ne, c.shexp_ff],
        "ffn_down_shexp.weight" if moe_layer => vec![c.shexp_ff, ne],
        // DeepSeek MLA's query LoRA, joint KV down-projection and output projection.
        "attn_q_a.weight" if mla => vec![ne, c.q_lora_rank],
        "attn_q_a_norm.weight" if mla => vec![c.q_lora_rank],
        "attn_q_b.weight" if mla => vec![c.q_lora_rank, c.n_head * mla_q_head_dim(c)],
        "attn_q.weight" if mla => vec![ne, c.n_head * mla_q_head_dim(c)],
        "attn_kv_a_mqa.weight" if mla => vec![ne, c.kv_lora_rank + c.qk_rope_dim],
        "attn_kv_a_norm.weight" if mla => vec![c.kv_lora_rank],
        "attn_output.weight" if mla => vec![c.n_head * c.v_head_dim, ne],
        _ => return None,
    };
    Some(shape)
}

/// One MLA query head: the no-RoPE part plus the RoPE part.
fn mla_q_head_dim(c: &Config) -> usize {
    c.head_k_mla + c.qk_rope_dim
}

/// Refuse `name` when its GGUF `shape` differs from [`expected_weight_shape`]. Trailing unit dims
/// are ignored on both sides, as GGML does: `[n]` and `[n, 1]` are the same tensor.
pub(super) fn check_weight_shape(c: &Config, name: &str, shape: &[usize]) -> AResult<()> {
    let Some(want) = expected_weight_shape(c, name) else {
        return Ok(());
    };
    let trim = |s: &[usize]| -> Vec<usize> {
        let keep = s.iter().rposition(|&d| d != 1).map_or(0, |i| i + 1);
        s[..keep].to_vec()
    };
    if trim(shape) != trim(&want) {
        return Err(anyhow!(
            "weight {name} has shape {shape:?}, but this model's metadata implies {want:?}"
        ));
    }
    Ok(())
}
