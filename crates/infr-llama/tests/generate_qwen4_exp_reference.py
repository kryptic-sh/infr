"""Regenerate the network-free f32 Qwen4-Exp causal oracle.

Requires CPU torch, numpy, safetensors and Transformers from revision
cc1451df3b5b218295bc16803569256c53c9d6e2. No GPU or remote model is used.
Usage: python tests/generate_qwen4_exp_reference.py --work-dir /tmp/qwen4-oracle
The checkpoint is generated from the committed tiny schema, not downloaded.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpForConditionalGeneration

REVISION = "cc1451df3b5b218295bc16803569256c53c9d6e2"
FIXTURES = Path(__file__).parent / "fixtures" / "qwen4_exp"
# Avoid unspecified top-k boundary ties in the parity fixture. Rust deliberately
# chooses the lowest block/expert ID on equal scores; HF topk is not stable.
TOKENS = [1, 17, 23, 2, 31, 7, 0, 48, 5, 79, 3, 98, 67, 122, 255, 8, 131]


def value(tensor, element, name):
    mask = (1 << 32) - 1
    v = (((tensor + 1) * 0x9E3779B9) & mask) ^ (((element + 1) * 0x85EBCA6B) & mask)
    v ^= v >> 16
    v = (v * 0x7FEB352D) & mask
    v ^= v >> 15
    result = ((v % 2049) - 1024) / 32768.0
    # Direct GDN output weights, unlike the additive Qwen RMSNorm parameters.
    return result + (1.0 if name.endswith("linear_attn.norm.weight") else 0.0)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--work-dir", type=Path, required=True)
    args.add_argument("--checkpoint", type=Path, help="Optional local real checkpoint instead of synthetic weights")
    args.add_argument("--output", type=Path, default=FIXTURES / "cpu_reference.json")
    a = args.parse_args()
    torch.set_num_threads(1)
    if a.checkpoint is None:
        a.work_dir.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((FIXTURES / "tensors.json").read_text())
        tensors = {}
        for ordinal, (name, spec) in enumerate(sorted(manifest.items())):
            shape = spec["shape"]
            if spec["dtype"] == "I64":
                tensor = torch.tensor(spec["values"], dtype=torch.int64)
            else:
                count = 1
                for dim in shape:
                    count *= dim
                tensor = torch.tensor([value(ordinal, j, name) for j in range(count)], dtype=torch.float32)
                tensor = tensor.to({"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[spec["dtype"]])
            tensors[name] = tensor.reshape(shape)
        save_file(tensors, a.work_dir / "model.safetensors")
        (a.work_dir / "config.json").write_bytes((FIXTURES / "config.json").read_bytes())
    checkpoint = a.checkpoint or a.work_dir
    model = Qwen4ExpForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.float32, attn_implementation="eager").eval()
    ids = torch.tensor([TOKENS])
    with torch.no_grad():
        whole = model.model.language_model(input_ids=ids, use_cache=True)
        whole_logits = model.lm_head(whole.last_hidden_state)[0].tolist()
        cache = None
        steps = []
        for position, token in enumerate(TOKENS):
            out = model.model.language_model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            step = {
                "hidden": out.last_hidden_state[0, 0].tolist(),
                "logits": model.lm_head(out.last_hidden_state)[0, 0].tolist(),
            }
            if position in (0, 3, 11, 16):
                step["gdn_state_0"] = cache.layers[0].recurrent_states[0].flatten().tolist()
                # HF stores channel-major K taps including current; the minimal
                # Rust state stores only the last K-1 raw projections, time-major.
                step["gdn_raw_0"] = cache.layers[0].conv_states[0][0, :, 1:].transpose(0, 1).flatten().tolist()
            steps.append(step)
    result = {"transformers_revision": REVISION, "tokens": TOKENS,
              "weight_pattern": "mix32-bf16-v1" if a.checkpoint is None else "local-real-checkpoint",
              "dtype": "float32", "steps": steps, "prefill_logits": whole_logits}
    a.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print("wrote", a.output, "bytes", a.output.stat().st_size)


if __name__ == "__main__":
    main()
