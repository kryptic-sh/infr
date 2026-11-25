## What this changes

<!-- What changed and why. One change per pull request, based on current main.
If it is stacked on another pull request, name it. -->

## How it was checked

<!-- The commands you ran and what they printed. GPU tests are ignored in CI:
list the ignored tests you ran and the device. For a performance change give
the device, driver, command, number of runs (builds alternated) and what you
could not measure. -->

## Checklist

The reasons are in `AGENTS.md`, "Before opening a pull request".

- [ ] Commit subjects are Conventional Commits of at most 72 characters, with
      bodies wrapped at 72 columns
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]`; a changed public API is
      under `Breaking`
- [ ] Docs follow the change: `docs/config.md` and `infr.example.toml` for a
      config key, the README for a server route or request field, the tuning
      docs for a tuning parameter
- [ ] A fix has a test that fails without it, and I saw it fail
- [ ] A performance change shows its output is unchanged, or no further from the
      CPU reference than before
- [ ] New GPU tests use tolerance testing against a reference (the tolerance
      harness, or a host reference for one op), not a stored hash or golden
      value
- [ ] Model math is compared with llama.cpp on the same input
- [ ] A server feature is tested against a real small model
- [ ] Every site with the same defect is fixed, and shared code is reused
      instead of copied
- [ ] Comments say why; measurements are here and in the changelog, not in code
- [ ] No new dependency without an agreed issue, and it is an optional feature
- [ ] New model work goes through the seam (CPU, Vulkan and Metal)
- [ ] `cargo fmt --all -- --check`,
      `cargo clippy --workspace --all-targets --locked -- -D warnings`,
      `cargo nextest run --workspace --locked` and
      `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked` pass
