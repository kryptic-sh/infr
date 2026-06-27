#!/usr/bin/env bash
# Apples-to-apples compare: infr (this engine, Vulkan) vs llama.cpp (Vulkan), SAME model + GPU.
#
# Both sides use the identical -p/-n/-d interface and emit `[{"avg_ts":X}]`:
#   • infr:   `infr bench`     (our CLI subcommand)
#   • llama:  `llama-bench`    (pacman: llama-cpp-vulkan), -fa auto (flash, as ollama runs)
# Pinned to the 7900 XTX (Vulkan0). pp = process an N-token prompt; tg = decode at context depth.
# Needs: llama-cpp-vulkan, bc, python3. Run from repo root: scripts/compare.sh [model.gguf] [tok.json]
set -euo pipefail

MODEL="${1:-${INFR_MODEL:-/home/mxaddict/Projects/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}}"
TOK="${2:-${INFR_TOK:-/home/mxaddict/Projects/models/qwen3-0.6b/tokenizer.json}}"
DEV="${INFR_CMP_DEV:-Vulkan0}"   # llama-bench device (the dGPU); override if device order differs
REPS="${INFR_CMP_REPS:-3}"
UB="${INFR_CMP_UB:-0}"           # ubatch (per-forward chunk). 0 = each tool's own default; pin to compare apples-to-apples
PP=(512 4096 8000 16000 32000)   # prefill prompt lengths
TG=(512 4096 8000 16000 32000)   # decode context depths

command -v llama-bench >/dev/null || { echo "llama-bench not found (pacman -S llama-cpp-vulkan)"; exit 1; }
command -v bc >/dev/null || { echo "bc not found (pacman -S bc)"; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }

echo "building infr..."
cargo build -q -p infr-cli --release
INFR=target/release/infr

avg_ts() { python3 -c "import sys,json; print(f\"{json.load(sys.stdin)[0]['avg_ts']:.0f}\")"; }
UB_I=(); UB_L=(); [ "$UB" -gt 0 ] && { UB_I=(-u "$UB"); UB_L=(-ub "$UB"); }
infr_b()  { "$INFR" bench "$MODEL" "${UB_I[@]}" "$@" --json 2>/dev/null | avg_ts; }
llama_b() { llama-bench -m "$MODEL" -ngl 99 -dev "$DEV" -fa auto -r "$REPS" "${UB_L[@]}" -o json "$@" 2>/dev/null | avg_ts; }
row() { printf '%-8s | %10s | %10s | %.2fx\n' "$1" "$2" "$3" "$(echo "scale=4; $2/$3" | bc)"; }

printf '\nmodel: %s   reps: %s   ubatch: %s\n' "$(basename "$MODEL")" "$REPS" "$([ "$UB" -gt 0 ] && echo "$UB" || echo "tool-default")"
printf '\n%-8s | %10s | %10s | %s\n' "PREFILL" "infr" "llama.cpp" "infr/llama"
printf -- '---------+------------+------------+-----------\n'
for n in "${PP[@]}"; do row "$n" "$(infr_b -p "$n" -n 0 -r "$REPS")" "$(llama_b -p "$n" -n 0)"; done

printf '\n%-8s | %10s | %10s | %s\n' "DECODE@d" "infr" "llama.cpp" "infr/llama"
printf -- '---------+------------+------------+-----------\n'
for d in "${TG[@]}"; do row "$d" "$(infr_b -p 0 -n 128 -d "$d" -r "$REPS")" "$(llama_b -p 0 -n 128 -d "$d")"; done
