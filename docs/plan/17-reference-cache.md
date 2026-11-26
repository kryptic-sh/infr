# 17 — Cache the CPU reference

**Why.** Every test and gate that scored a GPU or CPU-production run against the
CPU reference computed the reference again: there was no cache. The reference is
the same for every GPU, lane width and tuning mode of one commit, so the
tolerance gate paid for it once per run instead of once per commit. The owner
raised this on 2026-11-26 as the next slice after the PR #112 follow-up, because
it speeds our own work and every contributor's.

Slice 1 is built: the cache (`infr_llama::tolerance::oracle`), the gate wiring
(`gate::check_cached`, `Reference::cached`) and the timing line. How it works,
how to run and clear it, and what it never serves are in
[`docs/gpu-tolerance.md`](../gpu-tolerance.md), "The reference cache"; the
cleanup rules and the key are in the module's own doc. Figures marked **guess**
are guesses. The one measured split: on ERNIE-21B-A3B the reference took 3.9 to
10.5 s a prompt against 1.1 to 3.2 s for each GPU configuration.

Choices slice 1 made that the next slices inherit:

- The reference case of a gate run is the model file loaded again under the
  shipped defaults (`forced::reference_case`), under every mode including `off`,
  so the key does not split by device or mode.
- Rule 4 of the cleanup (a foreign magic or version) removes a file whose name
  is the cache's and whose magic is of the cache's family (`INFRTR`) in another
  version, or whose envelope version differs; a file of another family under one
  of the cache's names is not provably the cache's and is left alone. Temp files
  a dead writer left count as entries for rules 2 and 3.
- The code fingerprint is not a line of an entry's key: it names the generation
  directory, whose marker holds it whole and refuses an open with another one.
  It hashes every `CARGO_CFG_*` value besides the profile, and a test holds its
  crate list to infr-llama's workspace dependencies.
- A cache that cannot open for an environmental reason runs the gate without it,
  saying so in one line; a missing fingerprint fails the run only when a mode
  was asked for by name.
- The canary under `on` recomputes the shortest prompt among a case's greedy
  runs that were served; when none was, the first other run served (a KV-format
  or forced-along one) is recomputed instead.

## Slices

2. **The remaining callers and the measurement:** the gate's wall time on both
   GPUs, cache off against cold against warm, alternated, in a quiet window.
3. **Candidate, after slice 1's timings:** every forced run and every greedy
   generation opens a fresh Vulkan backend and uploads the weights again
   (ERNIE-21B's first GPU run took 6.4 s against 1.1 s after). Sharing one
   backend across the prompts of one configuration may be the larger saving on
   slow GPUs; it changes the isolation the harness relies on, so it needs its
   own bit-identity check.
