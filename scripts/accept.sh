#!/usr/bin/env bash
# Long-prompt acceptance for the KV modes: two conversations plus a continuation on a
# two-slot server, one run per mode, then a cross-mode diff of the answers.
#
#   scripts/accept.sh [linear|paged|store|both|all]
#
# The two distinct prompts matter: the second conversation is what lands on the second
# slot (the second page block under kv.paged), and that is where a KV write that ignores
# `row_base` shows up as garbled text. Greedy sampling, so every mode must answer the
# same thing.
set -euo pipefail

mode="${1:-all}"
root=$(cd "$(dirname "$0")/.." && pwd)
bin="$root/target/release/infr"
model="${INFR_MODEL:-unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL}"
port="${PORT:-8899}"
ctx="${CTX:-8k}"
work="$root/.cache/accept"

[ -x "$bin" ] || { echo "missing $bin — run: just build" >&2; exit 1; }
for f in a b; do
    [ -f "$root/fixtures/wikitext/part-$f.txt" ] || { echo "missing fixtures — run: just fixtures" >&2; exit 1; }
done
if pgrep -f "cargo/bin/infr serve|target/release/infr serve" >/dev/null; then
    echo "an infr server is already running; stop it first (one server per GPU)" >&2
    exit 1
fi

mkdir -p "$work"

flags_for() {
    case "$1" in
        linear) echo "--set kv.paged=false --set kv.store_mb=0" ;;
        paged)  echo "--set kv.paged=true  --set kv.store_mb=0" ;;
        store)  echo "--set kv.paged=false --set kv.store_mb=256" ;;
        both)   echo "--set kv.paged=true  --set kv.store_mb=256" ;;
        *) echo "unknown mode $1 (linear|paged|store|both|all)" >&2; exit 2 ;;
    esac
}

run_mode() {
    local m="$1"
    local log="$work/$m.log"
    # shellcheck disable=SC2046
    "$bin" serve "$model" --addr "127.0.0.1:$port" --ctx "$ctx" -n 2 --no-think \
        --temp 0 $(flags_for "$m") ${EXTRA:-} >"$log" 2>&1 &
    local pid=$!
    trap 'kill $pid 2>/dev/null || true' EXIT
    for _ in $(seq 1 300); do
        grep -q listening "$log" 2>/dev/null && break
        kill -0 "$pid" 2>/dev/null || { echo "server died:"; tail -5 "$log"; exit 1; }
        sleep 1
    done
    python3 - "$root" "$port" "$work/$m.json" <<'PY'
import json, sys, time, urllib.request

root, port, out = sys.argv[1], sys.argv[2], sys.argv[3]
base = "http://127.0.0.1:%s/v1/chat/completions" % port

def fixture(name):
    with open("%s/fixtures/wikitext/part-%s.txt" % (root, name), encoding="utf-8") as f:
        return f.read().strip()

def ask(text, question, prior=None):
    msgs = [{"role": "system", "content": text}]
    if prior:
        msgs += prior
    msgs.append({"role": "user", "content": question})
    body = json.dumps({"model": "infr", "messages": msgs, "temperature": 0, "max_tokens": 48,
                       "stream": False}).encode()
    req = urllib.request.Request(base, data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    text = ((msg.get("reasoning_content") or "") + (msg.get("content") or "")).strip()
    return {"text": text, "ms": int(dt * 1000), "usage": d.get("usage", {}),
            "finish": d["choices"][0].get("finish_reason")}

a, b = fixture("a"), fixture("b")
q = "In two sentences, what is the most notable fact in the text above?"
res = {}
res["a"] = ask(a, q)
# The same conversation continued: prefix reuse inside slot 0.
res["a2"] = ask(a, "Now name one person from the text.", prior=[
    {"role": "user", "content": q}, {"role": "assistant", "content": res["a"]["text"]}])
# A different conversation: this one lands on slot 1.
res["b"] = ask(b, q)
with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
for k in ("a", "a2", "b"):
    r = res[k]
    u = r["usage"]
    print("%s  prompt=%s tok  %d ms  %r" % (k, u.get("prompt_tokens"), r["ms"], r["text"][:110]))
PY
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    trap - EXIT
    sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -E "kv\.paged: [0-9]+ of|forked slot shares" | sed 's/^/    /' || true
}

if [ "$mode" = all ]; then
    for m in linear paged store both; do
        echo "== $m"
        run_mode "$m"
    done
    python3 - "$work" <<'PY'
import json, sys
work = sys.argv[1]
modes = ["linear", "paged", "store", "both"]
data = {m: json.load(open("%s/%s.json" % (work, m))) for m in modes}
print("\n== answers")
ref = data["linear"]
for conv in ("a", "a2", "b"):
    print(conv)
    for m in modes:
        r = data[m][conv]
        same = "=" if r["text"] == ref[conv]["text"] else "!"
        print("  %s %-6s prompt=%s %5d ms %r" % (same, m, r["usage"].get("prompt_tokens"), r["ms"], r["text"][:90]))
    if any(data[m][conv]["text"] != ref[conv]["text"] for m in modes):
        print("  ^ DIFFERS between modes")
PY
else
    echo "== $mode"
    run_mode "$mode"
fi
