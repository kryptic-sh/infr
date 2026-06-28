# Qwen3.5 / Qwen3.6 (`qwen35` / Qwen3-Next) support

Both `unsloth/Qwen3.5-*` and `unsloth/Qwen3.6-*` GGUFs declare
`general.architecture = qwen35`. It is the **Qwen3-Next** architecture: a hybrid of
**gated DeltaNet linear-attention** layers and **gated full-attention** layers, with SwiGLU
(dense for ≤1B; MoE for larger) and sectioned RoPE. Our engine is a pure transformer, so this
is a net-new architecture family. (Qwen3.6 has no ≤1B variant; smallest is 27B. Start with
`unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf` — already pulled.)

## 0.8B config (metadata)
- block_count=24, embedding_length=1024, context_length=262144
- full_attention_interval=4 → attention layers at i where (i+1)%4==0 → **layers 3,7,11,15,19,23**
  (6 attention); the other **18 are linear (gated DeltaNet)**.
- ssm: conv_kernel=4, state_size=128, inner_size=2048, group_count=16, time_step_rank=16
- attention: head_count=8, head_count_kv=2, key_length=value_length=256
- rope: dimension_count=64, dimension_sections=[11,11,10,0], freq_base=1e7
- rms_eps=1e-6

### Derived dims (llama.cpp reuses SSM hparam fields)
- head_k_dim = state_size = 128 ; num_k_heads = group_count = 16 → key_dim = 2048
- num_v_heads = time_step_rank = 16 ; d_inner = inner_size = 2048 → head_v_dim = 128
- conv_channels = d_inner + 2*group_count*state_size = 2048 + 4096 = **6144** (= attn_qkv out)

## Linear layer (gated DeltaNet) — per layer tensors
- attn_norm[1024] (input RMSNorm)
- attn_qkv[1024,6144] → q(2048)+k(2048)+v(2048)   (NO z here)
- attn_gate[1024,2048] → z (the output gate)
- ssm_conv1d[4,6144] depthwise causal conv over the 6144 qkv channels
- ssm_beta[1024,16] → b ; ssm_alpha[1024,16] → a   (per v-head)
- ssm_a[16] = −exp(A_log) ; ssm_dt.bias[16]
- ssm_norm[128] (RMSNorm over head_v_dim)
- ssm_out[2048,1024]
- ffn_norm? (post_attention_norm[1024]) + ffn_gate/up[1024,3584] + ffn_down[3584,1024]

### Forward (per token, decode/recurrent form — matches HF `fused_recurrent_gated_delta_rule`)
1. x = rmsnorm(hidden, attn_norm)
2. qkv = x @ attn_qkv  → [6144]; z = x @ attn_gate → [2048]
3. qkv = silu(causal_conv1d(qkv, ssm_conv1d, k=4))   (per-channel depthwise; state = last k-1 cols)
4. split qkv → q[16×128], k[16×128], v[16×128]
5. b = x@ssm_beta[16]; a = x@ssm_alpha[16]; beta = sigmoid(b);
   g = ssm_a * softplus(a + ssm_dt.bias)              (g ≤ 0, per v-head)
6. per head h (q,k L2-normed, scale=1/sqrt(128)):
   S = S * exp(g_h)                                   (S: [head_k_dim=128, head_v_dim=128])
   kv = kᵀS ; delta = (v − kv) * beta_h ; S += k ⊗ delta ; out_h = qᵀS   (out [128])
7. out = silu_gated_rmsnorm(out, ssm_norm, gate=z): rmsnorm(out)*silu(z)   (per head_v_dim)
8. y = out @ ssm_out → [1024]; hidden += y
9. h2 = rmsnorm(hidden, post_attention_norm); hidden += swiglu_ffn(h2)
   (note: confirm whether attn_norm/ffn ordering uses pre+post norms — see graph build_layer)

## Attention layer — per layer tensors
- attn_norm[1024]; attn_q[1024,4096]=q(8×256)+gate(2048); attn_k[1024,512]=2×256; attn_v[1024,512]
- attn_q_norm[256], attn_k_norm[256] (per-head RMSNorm); attn_output[2048,1024]
- ffn_* same as above
### Forward: standard GQA softmax attention, head_dim=256, 8 q / 2 kv heads, sectioned RoPE
  (dimension_sections [11,11,10,0]); output gated by sigmoid(gate) before attn_output.

## State caches (decode)
- conv state: per linear layer, [conv_channels=6144, k-1=3] rolling window.
- recurrent state: per linear layer, [num_v_heads=16, head_k_dim=128, head_v_dim=128] f32.
- KV cache: only for the 6 attention layers (head_dim 256).

## Build plan
1. CPU reference forward (loads real 0.8B, matches llama.cpp logits) — locks all math above.
2. GGUF loader: arch=qwen35, parse config, load both layer types, dequant new tensor dtypes.
3. GPU kernels: gated-delta recurrence (decode) + chunked form (prefill), gated attn hd=256,
   sectioned RoPE, silu-gated rmsnorm, causal conv1d; conv+recurrent state caches; hybrid routing.
4. head_dim=256 attention: our flash/attn kernels are hd=128-specialized → generalize or non-flash.

## References (read for math, reimplement — both MIT)
- HF transformers `modeling_qwen3_next` (recurrence: fused_recurrent + chunk_gated_delta_rule)
- llama.cpp `src/models/qwen3next.cpp` (GGUF tensor→role mapping, ggml_ssm_conv usage)
