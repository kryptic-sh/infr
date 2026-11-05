# Workflow recipes for this tree. `just --list` shows them; the shell is bash.
set shell := ["bash", "-euo", "pipefail", "-c"]

# Recipes, grouped: build, acceptance, tests, fixtures.
default:
    @just --list

# Release binary the acceptance runs use (scripts/accept.sh).
build:
    cargo build --release -p infr-cli

# Wikitext fixtures for long-prompt runs (committed already; only for regeneration).
fixtures:
    ./scripts/fixtures.sh

# Paged-KV tests: the graph-level parity cases plus the Vulkan attention kernels.
test-paged:
    cargo test -p infr-llama --test kv_paged_graph
    cargo test -p infr-vulkan --test attn_paged_pool --test attn_decode_parity --test kv_q8_roundtrip

# The whole workspace suite (long: the goldens touch real models).
test:
    cargo test --workspace

# Format and lint the way CI does.
check:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets
    RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links -D rustdoc::private_intra_doc_links" cargo doc --workspace --no-deps

# Two long conversations plus a continuation on a two-slot server, one mode at a time:
# `linear` (baseline), `paged` (page pool), `store` (host-RAM prefix store), `both`, or
# `all` to run every mode and diff the answers.
accept mode="all":
    ./scripts/accept.sh {{mode}}

# One server, for poking at by hand: `just serve paged` (mode: linear|paged|store|both).
serve mode="paged" port="8899":
    ./target/release/infr serve ${INFR_MODEL:-unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL} --addr 127.0.0.1:{{port}} --ctx ${CTX:-8k} -n 2 --no-think $(case {{mode}} in linear) echo --set kv.paged=false --set kv.store_mb=0;; paged) echo --set kv.paged=true --set kv.store_mb=0;; store) echo --set kv.paged=false --set kv.store_mb=256;; both) echo --set kv.paged=true --set kv.store_mb=256;; esac)
