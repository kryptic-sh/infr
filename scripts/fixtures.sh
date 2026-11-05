#!/usr/bin/env bash
# Regenerates fixtures/wikitext/part-*.txt from wikitext-2-raw-v1.
#
# The fixtures are committed, so this only needs to run when a fixture changes. The
# source is the same wikitext-2-raw-v1 archive llama.cpp's CI uses
# (huggingface.co/datasets/ggml-org/ci). Long plain-prose prompts are what the KV
# acceptance runs need: a short prompt never exercises prefill, a prefix share, or a
# second slot's page block.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cache="${INFR_FIXTURE_CACHE:-$root/.cache/fixtures}"
out="$root/fixtures/wikitext"
url="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

parts=${PARTS:-4}
words=${WORDS_PER_PART:-1500}
start_line=${START_LINE:-1}

mkdir -p "$cache" "$out"
if [ ! -f "$cache/wikitext-2-raw-v1.zip" ]; then
    echo "fetching $url"
    curl -fsSL -o "$cache/wikitext-2-raw-v1.zip" "$url"
fi
if [ ! -f "$cache/wiki.test.raw" ]; then
    (cd "$cache" && unzip -o -q wikitext-2-raw-v1.zip \
        && mv wikitext-2-raw/*.raw . && rmdir wikitext-2-raw)
fi

python3 - "$cache/wiki.test.raw" "$out" "$parts" "$words" "$start_line" <<'PY'
import os, sys

src, out, parts, words, start = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

def sentences(text):
    # wikitext keeps its own tokenisation: "@-@" is a hyphen split, " <unk>" a rare
    # token. Undo the first, drop the second, so the prose reads normally.
    text = text.replace(" @-@ ", "-").replace(" @-@", "-").replace("@-@ ", "-")
    return text.replace("<unk>", "").split()

with open(src, encoding="utf-8") as f:
    lines = f.read().split("\n")

pos = start
for i in range(parts):
    while pos < len(lines) and not lines[pos].strip():
        pos += 1
    chunk, count = [], 0
    while pos < len(lines) and count < words:
        line = lines[pos].strip()
        pos += 1
        if not line or line.startswith("=") or line == " ":
            continue
        s = sentences(line)
        if not s:
            continue
        chunk.append(" ".join(s))
        count += len(s)
    name = os.path.join(out, "part-%s.txt" % "abcd"[i % 4] + ("" if i < 4 else str(i)))
    with open(name, "w", encoding="utf-8") as g:
        g.write("\n\n".join(chunk) + "\n")
    print("%s  %d words" % (name, count))
PY
