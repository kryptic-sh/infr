# `infr-plat` — the platform seam

**Landed.** This file is what remains binding after the plan shipped; the design
argument that produced it is in the commit range `577328e..be1870c`.

## The rule

`infr-plat` owns every place the workspace touches an operating system directly.
Nothing else may depend on `libc` or on the `windows` crate.

That is enforced, not asked for: `crates/infr-plat/tests/platform_seam.rs` reads
every crate's manifest and fails on a new edge, naming the crate. There is one
allow-listed exception, `infr-vulkan`, with its reason in the list — see B69 in
[backlog.md](backlog.md).

New platform work goes in a module here, with an arm per platform and a test
that runs on the dev box. The crate is a **leaf**: it depends on no other
`infr-*` crate, which is why `signal::install_handlers` takes its shutdown latch
as a `fn(i32) -> bool` instead of calling `infr_core::shutdown` directly. Keep
it that way.

## What it owns

| module   | capability                                    |
| -------- | --------------------------------------------- |
| `fileio` | positioned reads and writes                   |
| `link`   | the content-addressed store's link step       |
| `lock`   | an exclusive, crash-safe file lock            |
| `mem`    | how much host memory could be committed       |
| `paths`  | config / cache / home directory resolution    |
| `proc`   | process liveness                              |
| `signal` | `SIGINT` / `SIGTERM`, or the local equivalent |
| `stdin`  | an interruptible byte read from fd 0          |

## What must not be sanded off

The recurring failure here is giving two different guarantees one signature, so
that "not supported on this platform" arrives at a caller as "passed". Three
cases are live, and each is documented at its definition rather than smoothed
over:

- **`mem::available`** is clamped by the cgroup limit on Linux, unclamped on
  Windows, and **absent on macOS**. `Available::source` carries which, because
  sizing an arena from an unclamped figure inside a container is an OOM kill.
- **`lock::FileLock`** is advisory on Unix (`flock`) and mandatory on Windows
  (`LockFileEx`). It promises the weaker of the two; a caller written against
  the stronger one would work on Windows and quietly do nothing on Linux.
- **`proc::pid_alive`** answers `true` where there is no probe — losing a
  tripwire rather than reporting a live process as dead. A new platform should
  get a real probe, not inherit that arm.

`fileio::write_all_at` is deliberately **not** in that list: std's unix
`write_all_at` loops internally over short writes and `EINTR`, and Windows has
no such wrapper, so the hand-rolled loop is the same operation. Classifying it
correctly matters as much as flagging the others.

## What is deliberately outside it

- **`infr-metal`.** It is a backend, and the hardware seam in this tree is
  `infr_core::backend::Backend`. Off macOS it already compiles to an empty lib
  with no dependencies. Whether Metal should be gated on a feature rather than
  on the host OS is B68.
- **`infr-vulkan`'s `p2p.rs` / `tp_sem.rs`.** B69.
- **`memmap2::Advice`** in `infr-gguf`, and the unix-only tests in
  `infr-core/src/kernel_cache.rs`. Neither names an OS library, so neither is
  what this rule is about.
