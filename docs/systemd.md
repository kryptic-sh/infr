# systemd socket activation

On Linux, systemd can keep infr's TCP port open while the inference process is
stopped. A new connection starts `infr serve`; the first request waits for the
model to load. After an idle timeout, infr exits successfully and releases its
model memory. The socket stays open for the next connection.

## Install user units

The templates in [`pkg/systemd`](../pkg/systemd/) run as your user and listen only
on `127.0.0.1:8080`. Install infr and pull your model before enabling the socket:

```bash
infr pull YOUR_MODEL_ID
mkdir -p ~/.config/systemd/user
cp pkg/systemd/infr.socket pkg/systemd/infr.service ~/.config/systemd/user/
```

Edit `~/.config/systemd/user/infr.service`: replace `REPLACE_WITH_MODEL_ID` with
your model ID or an absolute GGUF path, and change `%h/.local/bin/infr` to the
absolute path of your binary if needed. You can add normal serve options such as
`--ctx 8k` or `--parallel 2`. The service sets `INFR_SERVE_IDLE_SECS=300`; change
that value to choose the idle interval.

Use your usual `~/.config/infr/config.toml` for backend, device, and API-key
settings. The service does not inherit every environment variable from your
interactive shell; put required variables in the service's `Environment=` or an
`EnvironmentFile=`.

Then enable the **socket**, not the service:

```bash
systemctl --user daemon-reload
systemctl --user enable --now infr.socket
curl http://127.0.0.1:8080/health
systemctl --user status infr.socket infr.service
journalctl --user -u infr.service -f
```

Stop any existing server using the same port first. Change `ListenStream=` in
`infr.socket` to use another port; the inherited socket takes precedence over
`serve --addr`. Apply a port change by stopping the service and restarting the
socket after `daemon-reload`.

User units normally stop when the user manager exits. If the server must remain
available while you are logged out, enable lingering for your user:

```bash
loginctl enable-linger "$USER"
```

To stop serving, stop both units; stopping only the service leaves the socket
able to activate it again:

```bash
systemctl --user disable --now infr.socket
systemctl --user stop infr.service
```

## Idle behavior

`serve.idle_timeout_secs` / `INFR_SERVE_IDLE_SECS` is disabled by default (`0`).
It also works without systemd:

```bash
infr --set serve.idle_timeout_secs=300 serve YOUR_MODEL_ID
```

Without a socket supervisor, exiting on idle does not provide automatic wake-up.

The timer starts when serving begins and resets when the last request finishes.
Queued requests, image validation, generation, and response streams prevent idle
exit. Work still running after a client disconnect also prevents it. Idle exit
does not cancel generation or set the process-wide signal shutdown flag. Shutdown
also waits for HTTP connections to drain; a stalled response write can delay exit.

All HTTP requests count as activity, including health checks, authentication
failures, `/status`, and `/metrics`. Frequent probes can prevent idle exit and
wake a stopped service. An unused HTTP keep-alive connection does not keep the
process awake. Idle shutdown closes it; clients must reconnect. A request racing
with shutdown may receive HTTP 503 and should retry on a new connection.

The first request after activation pays model-loading time. Give clients enough
time to wait, especially for large models. systemd buffers new TCP connections;
it is not an HTTP reverse proxy and does not migrate established connections to
the next process.

## Unit requirements

- Use `Accept=no`: infr serves all requests through one shared listener and model.
- Pass exactly one listening TCP socket, IPv4 or IPv6. Multiple listeners and
  Unix-domain sockets are rejected; use a reverse proxy if you need them.
- `NonBlocking=yes` belongs in the service. infr also sets nonblocking mode when
  importing the listener.
- Use `Restart=on-failure`, not `always` or `on-success`. A clean idle exit must
  leave the process stopped until another connection arrives.
- Do not use `RuntimeMaxSec=` as an idle timer: it expires even during active work.

Socket inheritance is Linux-only. On other platforms, infr binds `--addr` as
usual; idle exit remains available.
