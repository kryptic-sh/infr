# Wikitext fixtures

Long plain-prose prompts for the KV acceptance runs (`just accept`). They are slices of
[wikitext-2-raw-v1](https://huggingface.co/datasets/ggml-org/ci), the same archive
llama.cpp's CI uses; the data is CC BY-SA 3.0 / GFDL, from Wikipedia.

Each file is ~1500 words (~1800-2000 tokens), long enough to make prefill measurable and to
cross page and prefix-block boundaries. `part-a` and `part-b` are two *different* documents on
purpose: the second one is the request that lands on the second KV slot (the second page block
under `kv.paged`).

Regenerate with `just fixtures` (needs network); the files are committed so the runs work
offline.
