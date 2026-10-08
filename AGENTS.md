# Repository instructions

## Shell and local tests

- Use Git Bash on Windows when available. Do not use PowerShell.
- Keep Windows builds and tests native to Windows; do not substitute WSL for
  Windows verification.
- Run all local tests and their subprocesses headlessly, without visible console
  windows. Use hidden or no-window process creation when a runner needs it.
- Use the integrated GPU for GPU testing and performance measurements. Prefer
  cached small models to limit disk usage, and run GPU workloads serially.

## Branch and delivery

- Keep all work on `main`.
- Pull the latest changes before continuing each slice of work, preserving local
  changes when integrating upstream work.
- Commit and push each completed slice to `main`.
- Watch CI for the pushed commit, inspect its jobs, and resolve failures before
  considering the slice complete. A successful push alone is not verification.
