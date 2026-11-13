# Repository instructions

## Shell and local tests

- Use Git Bash on Windows when available. Do not use PowerShell.
- Keep Windows builds and tests native to Windows; do not substitute WSL for
  Windows verification.
- Run all local tests and their subprocesses headlessly, without visible console
  windows. Use hidden or no-window process creation when a runner needs it.
- Both GPUs on the Windows box may be used: the integrated GPU and the RX 7900
  XTX. Pick the device with `INFR_DEV` after checking `infr devices`, say which
  one a result came from, and run GPU workloads serially.
- Prefer cached small models. Downloading a model to test with is allowed while
  disk space lasts; check the free space and the file size first.

## Branch and delivery

- Keep all work on `main`.
- Pull the latest changes before continuing each slice of work, preserving local
  changes when integrating upstream work.
- Commit and push each completed slice to `main`.
- Watch CI for the pushed commit, inspect its jobs, and resolve failures before
  considering the slice complete. A successful push alone is not verification. A
  push cancels the run before it, so let a run finish before pushing the next
  slice when its verdict is still owed.

## How work is run

- The main agent orchestrates. Each slice goes through a coder agent, then a
  separate reviewer agent that reads the diff without being told what to
  conclude; the orchestrator re-checks both reports against the tree before
  committing.
- While sub-agents are working, the orchestrator plans what comes next rather
  than waiting: plans under `docs/plan/`, findings in `docs/backlog.md`.
- When the backlog's actionable items run out, have a reviewer read a part of
  the codebase no earlier pass covered and record what it finds in the backlog.

## What to work on

- Correctness is the top priority, and performance matters just as much: a
  change that buys one must not quietly cost the other. Both come before
  anything else in improvement work.
- Do not shy away from a large refactor when it is worth it for correctness, for
  performance, or for both. Say what it buys, and verify it: a test that can
  fail for correctness, a measurement for performance.
- New model families and planned features are in scope. Support recent models
  people will actually run; an older or less useful family gets support only
  when it is genuinely little work. The reference for a port is the llama.cpp
  checkout at `~/Projects/mxaddict/llama.cpp`.
- Streaming is a standing goal: disk to DRAM to VRAM, so that any model size
  runs on any hardware. That includes the cache eviction logic in every tier,
  which should be as fast as it can be made.
