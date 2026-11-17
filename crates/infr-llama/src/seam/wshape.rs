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
/// read them in their own head-major layout), the V3.2 lightning indexer, the DeepSeek V4
/// `ffn_gate_tid2eid` routing table (`generate_dense_backend` checks it against the vocabulary
/// itself), the qwen35 DeltaNet tensors, the dual-FFN (gemma4 MoE / DiffusionGemma) router and
/// expert banks and their extra norms, the gemma4 E2B per-layer embedding tensors,
/// DiffusionGemma's self-conditioning MLP, and a KV-sharing layer's own `attn_k`/`attn_v` (loaded,
/// but never read by the graph).
///
/// The DeepSeek V4 rules transcribe `deepseek4.cpp::load_arch_tensors` at the head width the V4
/// graph indexes by, [`Config::head_dim`] (`attention.key_length`). llama.cpp sizes those tensors
/// with layer 0's head width instead, which on V4 — every layer sliding-window — is
/// `attention.key_length_swa` when a file declares it. No V4 file does; one that did, with tensors
/// shaped to match, is refused here by name rather than read at the wrong stride.
pub(super) fn expected_weight_shape(c: &Config, name: &str) -> Option<Vec<usize>> {
    let ne = c.n_embd;
    match name {
        "token_embd.weight" | "output.weight" => return Some(vec![ne, c.vocab]),
        "output_norm.weight" => return Some(vec![ne]),
        // V4's hyper-connection head, collapsing the `hc_mult` residual streams before
        // `output_norm`.
        "output_hc_fn.weight" if c.deepseek4 => return Some(vec![c.hc_mult * ne, c.hc_mult]),
        "output_hc_base.weight" if c.deepseek4 => return Some(vec![c.hc_mult]),
        "output_hc_scale.weight" if c.deepseek4 => return Some(vec![1]),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (l, role) = rest.split_once('.')?;
    let l: usize = l.parse().ok()?;
    if l >= c.n_layer {
        return None;
    }
    let delta = c.qwen35 && !c.is_qwen35_attn_layer(l);
    let shortconv = c.is_shortconv_layer(l);
    let mla = c.deepseek2 && !c.deepseek4;
    // The standard attention mixer (`MixerW::Attn`): every arch but DeltaNet, LFM's short
    // convolution, MLA and V4.
    let attn = !delta && !shortconv && !c.deepseek2 && !c.deepseek4;
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
        // LFM3's per-head output gate: one logit per head (`Op::HeadGate`).
        "attn_gate.weight" if attn && c.lfm3 => vec![ne, c.n_head],
        // LFM's short convolution: B | C | X in-projection, the per-channel kernel, the output.
        "shortconv.in_proj.weight" if shortconv => vec![ne, 3 * ne],
        "shortconv.conv.weight" if shortconv => vec![c.shortconv_l_cache, ne],
        "shortconv.out_proj.weight" if shortconv => vec![ne, ne],
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
        _ if c.deepseek4 => return dsv4_weight_shape(c, l, role),
        _ => return None,
    };
    Some(shape)
}

/// The DeepSeek V4 mixer, compressor, indexer and hyper-connection tensors of layer `l`.
fn dsv4_weight_shape(c: &Config, l: usize, role: &str) -> Option<Vec<usize>> {
    let (ne, hd, nh) = (c.n_embd, c.head_dim, c.n_head);
    let hc_mix = (2 + c.hc_mult) * c.hc_mult;
    let ratio = c.layer_compress_ratio(l);
    // The CSA (ratio-4) compressor writes two overlapping windows per block, so its projections
    // are twice as wide as the HCA (ratio-128) one's.
    let coff = if ratio == 4 { 2 } else { 1 };
    let (ixh, ixd) = (c.indexer_n_head, c.indexer_head_size);
    let shape = match role {
        "attn_sinks.weight" => vec![nh],
        "attn_q_a.weight" => vec![ne, c.q_lora_rank],
        "attn_q_a_norm.weight" => vec![c.q_lora_rank],
        "attn_q_b.weight" => vec![c.q_lora_rank, nh * hd],
        "attn_kv.weight" => vec![ne, hd],
        "attn_kv_a_norm.weight" => vec![hd],
        // Each of the `o_group_count` groups projects its own `n_head * head_dim / o_group_count`
        // slice of the concatenated heads down to `o_lora_rank`. A count that does not divide the
        // heads has no such slice; `generate_dense_backend` refuses that by name.
        "attn_output_a.weight" => {
            let rows = (nh * hd).checked_div(c.o_group_count)?;
            vec![rows, c.o_lora_rank * c.o_group_count]
        }
        "attn_output_b.weight" => vec![c.o_group_count * c.o_lora_rank, ne],
        "hc_attn_fn.weight" | "hc_ffn_fn.weight" => vec![c.hc_mult * ne, hc_mix],
        "hc_attn_base.weight" | "hc_ffn_base.weight" => vec![hc_mix],
        // One scale each for the `pre`, `post` and `comb` mixes.
        "hc_attn_scale.weight" | "hc_ffn_scale.weight" => vec![3],
        "attn_compressor_kv.weight" | "attn_compressor_gate.weight" if ratio != 0 => {
            vec![ne, coff * hd]
        }
        "attn_compressor_ape.weight" if ratio != 0 => vec![coff * hd, ratio],
        "attn_compressor_norm.weight" if ratio != 0 => vec![hd],
        "indexer.proj.weight" if ratio == 4 => vec![ne, ixh],
        "indexer.attn_q_b.weight" if ratio == 4 => vec![c.q_lora_rank, ixh * ixd],
        "indexer_compressor_kv.weight" | "indexer_compressor_gate.weight" if ratio == 4 => {
            vec![ne, coff * ixd]
        }
        "indexer_compressor_ape.weight" if ratio == 4 => vec![coff * ixd, ratio],
        "indexer_compressor_norm.weight" if ratio == 4 => vec![ixd],
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
///
/// V4's `attn_output_a` is compared with its dims past the first folded into one: llama.cpp loads
/// it as `[rows, o_lora_rank, o_group_count]` from a file that may store either that or the 2-D
/// `[rows, o_lora_rank * o_group_count]` (`TENSOR_ALLOW_RESHAPE`), and the two are the same bytes.
/// The row width is still checked exactly, since that is what the grouped projection strides by.
pub(super) fn check_weight_shape(c: &Config, name: &str, shape: &[usize]) -> AResult<()> {
    let Some(want) = expected_weight_shape(c, name) else {
        return Ok(());
    };
    let trim = |s: &[usize]| -> Vec<usize> {
        let keep = s.iter().rposition(|&d| d != 1).map_or(0, |i| i + 1);
        s[..keep].to_vec()
    };
    let folded;
    let got = match shape {
        [rows, rest @ ..] if c.deepseek4 && name.ends_with(".attn_output_a.weight") => {
            folded = [*rows, rest.iter().product()];
            &folded[..]
        }
        _ => shape,
    };
    if trim(got) != trim(&want) {
        return Err(anyhow!(
            "weight {name} has shape {shape:?}, but this model's metadata implies {want:?}"
        ));
    }
    Ok(())
}
