# 17 — Cache the CPU reference

**Why.** Every test and gate that scores a GPU or CPU-production run against the
CPU reference computes the reference again: there is no cache. The reference is
the same for every GPU, lane width and tuning mode of one commit, so the
tolerance gate pays for it once per run instead of once per commit. The owner
raised this on 2026-11-26 as the next slice after the PR #112 follow-up, because
it speeds our own work and every contributor's.

Nothing here is built. Figures marked **guess** are guesses. The one measured
split: on ERNIE-21B-A3B the reference took 3.9 to 10.5 s a prompt against 1.1 to
3.2 s for each GPU configuration. How much of the whole gate is reference time
is not measured; slice 1 adds the timing line that says.

## The danger it must rule out

A stale entry serving as the oracle: the CPU reference's code or inputs change,
the key does not, and tests pass or fail against an old answer. Every choice
below is made against that failure.

## What a comparison takes from the reference

`seam::forced_run` through `SeamModel::forced_logits` produces a `ForcedLogits`:
the logit rows, the positions, the prefill row count, the token sequence, and on
a model with routed experts the routing (the ids the GPU run is forced along,
and the scores). No KV state crosses over. That whole value is one cache entry.

## The key, stored verbatim and compared on read

- **Code:** a build-script hash, in `crates/infr-llama/build.rs`, over the
  content of the sources and manifests of every crate the CPU reference runs
  (`infr-core`, `infr-cpu`, `infr-gguf`, `infr-llama`, `infr-chat`, `infr-plat`,
  `infr-prof`, `infr-prof-rt`), the lockfile, the compiler version, target,
  profile, target features, flags and enabled features. A manually bumped
  constant fails silently when someone forgets; a git commit needs git at build
  time and is only as fine as its pathspec. `infr-vulkan` and `infr-metal` are
  left out on purpose: the reference runs none of their code, and hashing them
  would miss on every shader edit, the loop this cache exists for. A build with
  no sibling sources gets no fingerprint and the cache refuses to run.
- **Weights:** a content hash of every shard's bytes, in parallel chunks. Not
  the Hugging Face blob name (test-built and planted files are not in that
  cache) and not a file stamp. A model planted in memory is not cached.
- **Inputs:** the prompt ids, the step count, the prefill-head choice, how the
  next token is picked (greedy, or forced along given tokens), and a digest of
  any forced routing.
- **Configuration:** every path of the engine configuration as text. To keep
  that from splitting the cache by GPU mode, the reference runs under the
  shipped defaults plus the KV-format edit, as `infr tune check` already does,
  by passing the gate a separately loaded reference case.
- **Machine:** architecture and OS. The reference is bit-identical across thread
  counts and instruction-set tiers (plan 11), so those are not keyed.

About every other commit changes the hashed crates and starts cold; within one
commit every GPU and mode shares one reference.

## Format and place

- The envelope `KernelCache` already uses (magic, version, verbatim key,
  checksum, atomic durable write), lifted into shared functions with
  `KernelCache` rebased on them unchanged. A checksum or key mismatch is a miss;
  a checksum failure deletes the file.
- Under the user cache directory (`infr_plat::paths::cache_dir()`), not under
  `target/`: it survives `cargo clean` and is shared between worktrees whose
  hashed sources match. So the size cap and the clear command matter.
- Off when `CI` is set.
- About 2.2 GB a code generation for the gated models (**guess**, from vocab
  sizes and row counts).
- No lock on read or write: readers take one whole read, writers rename a
  finished temp file, and two writers of one key write the same bytes.

## Clearing old values

Entries go stale when the code fingerprint changes (a whole generation), a model
file is deleted or replaced, the tests stop using a prompt or token count, or
the format version moves. They are removed without a database:

1. Entries live in a directory per code fingerprint with a marker file and a
   last-used stamp. A generation other than the current one is removed when its
   stamp is older than an age limit (default three days), not the moment a new
   fingerprint writes, so two worktrees on different commits do not throw each
   other's cache away.
2. Entries of the current generation unused past the same age are removed; that
   covers deleted models and dropped prompts.
3. A total size cap (default 6 GB) evicts the least recently used across
   generations.
4. Files with a foreign magic or version are removed when their name matches.

Cleanup runs once per process when the cache opens, before any test timer, and
only if it wins a `FileLock`. It never deletes anything newer than an hour,
anything outside its root, a symlink, or a file it cannot prove is its own by
name pattern and header; it removes files one by one and a directory only when
empty. A clear mode removes everything it owns.

## Safety switches

- `INFR_TOL_REFCACHE` (test support only; the library takes an explicit cache):
  `off`, `on` (local default), `verify` (always recompute and assert bit
  equality with the stored entry), `refresh`, `clear`.
- **A canary under `on`:** each reference served from the cache recomputes its
  shortest prompt and compares bit for bit; a mismatch fails the run and names
  the entry. It catches a gap in the key or fingerprint.
- One log line per lookup (hit, miss, stored, verified) and a per-case line of
  reference seconds computed against read, so a report can say which reference
  it used.
- **Never read from the cache:** any test whose subject is the CPU reference or
  CPU production (`forced_run_reproduces_the_cpu_reference_on_every_path`, the
  self-checks, the planted demonstrations' subject runs, the `cpu_act_quant`
  tests, the CPU goldens, the int8 prefill accuracy test, the CPU production run
  of the measurement). Only the cached oracle functions read it; the plain
  reference functions never do.

## Where it hooks

A library module, `infr_llama::tolerance::oracle`, used by the gate's reference
(a new cached constructor beside the uncached one) at its three sites: the
greedy reference, the KV-format reference and the forced-along reference.
`infr tune check` stays uncached until the gain is measured; opting it in is one
line.

## Slices

1. **The cache, the gate wiring, the timing line.** Tests in the normal job, on
   synthetic models and temp directories with an injected fingerprint: a miss on
   one weight byte, one prompt token, the step count, the KV pair, one routing
   id, the code fingerprint, one config path; a hit on the same file at another
   path; a flipped byte rejected and deleted; verify and the canary each catch a
   tampered entry; two writers leave a readable entry; the cached result equals
   the uncached one bit for bit; each cleanup rule, including a foreign file and
   a symlink left alone. Docs: a section in `docs/gpu-tolerance.md` on running
   the tolerance tests with the cache, and a pointer in `AGENTS.md`.
2. **The remaining callers and the measurement:** the gate's wall time on both
   GPUs, cache off against cold against warm, alternated, in a quiet window.
3. **Candidate, after slice 1's timings:** every forced run and every greedy
   generation opens a fresh Vulkan backend and uploads the weights again
   (ERNIE-21B's first GPU run took 6.4 s against 1.1 s after). Sharing one
   backend across the prompts of one configuration may be the larger saving on
   slow GPUs; it changes the isolation the harness relies on, so it needs its
   own bit-identity check.
