//! DNS-rebinding protection: the `Host` check every route but `/health` passes
//! (`serve.allowed_hosts`).

use std::net::SocketAddr;

use axum::{
    extract::{Request, State},
    http::{uri::Authority, StatusCode},
    middleware::Next,
    response::Response,
};
use infr_core::config::Config;

use crate::{configured_api_key, invalid_request, AppState};

/// The loopback names a keyless loopback server answers to, whatever loopback address it is bound
/// to, beside every `*.localhost` name (RFC 6761 reserves those for loopback). IPv6 literals keep
/// the brackets a `Host` header spells them with.
const LOOPBACK_HOSTS: [&str; 3] = ["localhost", "127.0.0.1", "[::1]"];

/// The `Host` check every route but `/health` runs first: DNS-rebinding protection. A layer, like
/// the bearer check, so a refused request is answered on its headers alone.
pub(crate) async fn host_gate(State(state): State<AppState>, req: Request, next: Next) -> Response {
    // HTTP/2 carries the host as the URI authority instead of a `Host` header.
    let host = match req.headers().get(axum::http::header::HOST) {
        Some(v) => Some(v.to_str().unwrap_or_default()),
        None => req.uri().authority().map(Authority::as_str),
    };
    if host_allowed(&state.cfg, state.bind, host) {
        return next.run(req).await;
    }
    tracing::warn!(host = ?host, "refused a request for a host not in serve.allowed_hosts");
    invalid_request(
        StatusCode::FORBIDDEN,
        None,
        None,
        format!(
            "Host {:?} is not allowed. Add it to serve.allowed_hosts \
             (INFR_SERVE_ALLOWED_HOSTS), or set that to \"*\" to turn the Host check off.",
            host.unwrap_or_default()
        ),
    )
}

/// The configured `serve.allowed_hosts`, `None` when unset — or set to nothing, an empty value
/// or a list of only blanks, which every layer reads as "use the default policy".
fn allowed_hosts(cfg: &Config) -> Option<&[String]> {
    cfg.serve
        .allowed_hosts
        .as_deref()
        .filter(|list| list.iter().any(|h| !h.trim().is_empty()))
}

/// Refuse a `serve.allowed_hosts` entry that could never match a `Host` header — a URL, a path, a
/// wildcard pattern other than the bare `*` — at startup, instead of leaving it silently dead
/// while the requests it was meant to admit get `403`s.
pub(crate) fn check_allowed_hosts(cfg: &Config) -> anyhow::Result<()> {
    for entry in allowed_hosts(cfg).unwrap_or_default() {
        let entry = entry.trim();
        if entry.is_empty() || entry == "*" {
            continue;
        }
        if entry.contains('*') || split_host(entry).is_none() {
            anyhow::bail!(
                "serve.allowed_hosts entry {entry:?} is not a host name or address (with an \
                 optional port); write the bare name a client puts in its Host header, or `*` \
                 alone to turn the check off"
            );
        }
    }
    Ok(())
}

/// Whether a request naming `host` (`None`: no `Host` header at all) may proceed, for a server
/// bound to `bind`.
///
/// A web page whose hostname its owner re-resolves to `127.0.0.1` becomes same-origin with a
/// loopback server, and a keyless one would answer it — but the page's requests still carry the
/// page's hostname. So a keyless server bound to a loopback address answers only the loopback
/// names, `*.localhost`, and its own bind address. A server with an API key, or bound to a
/// non-loopback address, chose its exposure and is not checked — unless `serve.allowed_hosts` is
/// set, which always turns the check on and allows the listed names beside the loopback ones;
/// `*` turns it off.
///
/// Ports are not compared: the attacker picks the name a page resolves, never the port it
/// reaches — and a client behind `ssh -L` or a container port mapping names a port the server
/// did not bind. A request with no `Host` at all passes: a browser always sends one (and a
/// rebinding page cannot drop it), so its absence means an HTTP/1.0 tool, not the attack.
fn host_allowed(cfg: &Config, bind: Option<SocketAddr>, host: Option<&str>) -> bool {
    let listed = allowed_hosts(cfg);
    match listed {
        Some(list) if list.iter().any(|h| h.trim() == "*") => return true,
        Some(_) => {}
        None => {
            let exposed =
                configured_api_key(cfg).is_some() || !bind.is_some_and(|a| a.ip().is_loopback());
            if exposed {
                return true;
            }
        }
    }
    let Some(host) = host else {
        return true;
    };
    let Some((name, _)) = split_host(host) else {
        return false;
    };
    LOOPBACK_HOSTS.contains(&name.as_str())
        // RFC 6761 has resolvers answer `*.localhost` with loopback themselves. One that forwards
        // such a name to DNS instead would let a domain owner point it elsewhere — and rebind it
        // to this server; the check cannot see the resolver, so that residual stays.
        || name.ends_with(".localhost")
        || bind.is_some_and(|b| name == bind_host(b))
        || listed.is_some_and(|list| {
            list.iter()
                .filter_map(|h| split_host(h))
                .any(|(allowed, _)| allowed == name)
        })
}

/// `host[:port]` as a lowercased name without a trailing dot, and the port. `None` for anything
/// that is not a plain authority — a scheme, a path, userinfo (which a `Host` never carries).
fn split_host(host: &str) -> Option<(String, Option<u16>)> {
    if host.contains(['@', '/']) {
        return None;
    }
    let a: Authority = host.trim().parse().ok()?;
    let name = a.host().trim_end_matches('.').to_ascii_lowercase();
    (!name.is_empty()).then_some((name, a.port_u16()))
}

/// The bind address's IP as a `Host` header names it.
fn bind_host(bind: SocketAddr) -> String {
    match bind {
        SocketAddr::V4(a) => a.ip().to_string(),
        SocketAddr::V6(a) => format!("[{}]", a.ip()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_router;
    use axum::{body::Body, http::Request as HttpRequest};
    use http_body_util::BodyExt;
    use std::sync::Arc;
    use tower::ServiceExt;

    fn cfg(api_key: Option<&str>, allowed: Option<&[&str]>) -> Config {
        let mut c = Config::default();
        c.serve.api_key = api_key.map(str::to_owned);
        c.serve.allowed_hosts = allowed.map(|l| l.iter().map(|h| h.to_string()).collect());
        c
    }

    fn addr(s: &str) -> Option<SocketAddr> {
        Some(s.parse().unwrap())
    }

    /// The default policy guards a keyless loopback server: the loopback names, `*.localhost`
    /// and its own bind address pass on any port (a forwarded port is not the attack), as does a
    /// request with no `Host` (no browser sends one); every other name — the rebinding page's,
    /// or a non-loopback alias like `host.docker.internal` — is refused.
    #[test]
    fn a_keyless_loopback_server_answers_only_loopback_names() {
        let c = cfg(None, None);
        let bind = addr("127.0.0.1:8080");
        for ok in [
            "localhost",
            "localhost:8080",
            "localhost:2222",
            "LOCALHOST:8080",
            "localhost.:8080",
            "127.0.0.1:9999",
            "[::1]:8080",
            "[::1]",
            "app.localhost",
            "a.b.localhost:3000",
        ] {
            assert!(host_allowed(&c, bind, Some(ok)), "{ok} must pass");
        }
        assert!(host_allowed(&c, bind, None), "no Host header at all");
        for refused in [
            "evil.example",
            "evil.example:8080",
            "host.docker.internal",
            "localhost.evil.example",
            "127.0.0.2:8080",
            "user@localhost:8080",
            "",
        ] {
            assert!(
                !host_allowed(&c, bind, Some(refused)),
                "{refused} must be refused"
            );
        }

        // The bind address itself passes, here an IPv6 one, bracketed as a `Host` spells it.
        let v6 = addr("[::1]:9000");
        assert!(host_allowed(&c, v6, Some("[::1]:9000")));
        assert!(!host_allowed(&c, v6, Some("evil.example:9000")));
        assert!(host_allowed(
            &c,
            addr("127.0.0.5:8080"),
            Some("127.0.0.5:1")
        ));
    }

    /// A server with an API key, or bound off loopback, chose its exposure and is not checked —
    /// and neither is one whose bind address is unknown.
    #[test]
    fn exposed_servers_are_not_checked_by_default() {
        assert!(host_allowed(
            &cfg(Some("k"), None),
            addr("127.0.0.1:8080"),
            Some("evil.example")
        ));
        assert!(host_allowed(
            &cfg(None, None),
            addr("0.0.0.0:8080"),
            Some("evil.example")
        ));
        assert!(host_allowed(&cfg(None, None), None, Some("evil.example")));
        // An empty key is no key: still guarded.
        assert!(!host_allowed(
            &cfg(Some(""), None),
            addr("127.0.0.1:8080"),
            Some("evil.example")
        ));
    }

    /// An explicit `serve.allowed_hosts` always applies — key or not, loopback or not — and adds
    /// its names (any port) to the loopback ones; `*` switches the check off. A list with nothing
    /// in it is unset, not "check everywhere".
    #[test]
    fn an_explicit_allow_list_always_applies_and_star_disables_it() {
        let c = cfg(Some("k"), Some(&["llm.example.com", "Proxy.Internal:8443"]));
        for bind in [addr("0.0.0.0:8080"), addr("127.0.0.1:8080"), None] {
            assert!(host_allowed(&c, bind, Some("llm.example.com")));
            assert!(host_allowed(&c, bind, Some("llm.example.com:443")));
            assert!(host_allowed(&c, bind, Some("proxy.internal")));
            assert!(host_allowed(&c, bind, Some("localhost:8080")));
            assert!(!host_allowed(&c, bind, Some("evil.example")), "{bind:?}");
        }
        for blank in [&[][..], &[""][..], &["  ", ""][..]] {
            let unset = cfg(Some("k"), Some(blank));
            assert!(
                host_allowed(&unset, addr("0.0.0.0:8080"), Some("evil.example")),
                "{blank:?} must mean unset"
            );
        }
        let star = cfg(None, Some(&["*"]));
        assert!(host_allowed(
            &star,
            addr("127.0.0.1:8080"),
            Some("evil.example")
        ));
    }

    /// An entry that can never match a `Host` header is a startup error, not a dead entry.
    #[test]
    fn an_unmatchable_allowed_host_is_a_config_error() {
        for ok in [
            &["llm.example.com", "10.0.0.2:8000", "[::1]", "*"][..],
            &[""][..],
        ] {
            assert!(check_allowed_hosts(&cfg(None, Some(ok))).is_ok(), "{ok:?}");
        }
        assert!(check_allowed_hosts(&cfg(None, None)).is_ok());
        for bad in ["http://x", "x/path", "*.example.com", "user@x"] {
            let e = check_allowed_hosts(&cfg(None, Some(&[bad]))).unwrap_err();
            assert!(e.to_string().contains(bad), "{bad}: {e}");
        }
    }

    async fn get(router: axum::Router, uri: &str, host: &str) -> (StatusCode, String) {
        let resp = router
            .oneshot(
                HttpRequest::builder()
                    .uri(uri)
                    .header("host", host)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    /// Through the router: a rebinding page's request is a `403` whose JSON body names the key to
    /// set, a loopback request passes, and `/health` answers whatever the host.
    #[tokio::test]
    async fn a_refused_host_is_a_403_naming_the_key_and_health_stays_open() {
        let mut state = AppState::headless("m", Arc::new(Config::default()));
        state.bind = addr("127.0.0.1:8080");
        let router = build_router(state);

        for uri in ["/v1/models", "/status", "/metrics"] {
            let (status, body) = get(router.clone(), uri, "evil.example:8080").await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{uri}");
            let v: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v["error"]["type"], "invalid_request_error");
            let msg = v["error"]["message"].as_str().unwrap();
            assert!(msg.contains("serve.allowed_hosts"), "{msg}");
            assert!(msg.contains("evil.example:8080"), "{msg}");
        }
        let (status, _) = get(router.clone(), "/v1/models", "localhost:8080").await;
        assert_eq!(status, StatusCode::OK);
        let (status, _) = get(router, "/health", "evil.example:8080").await;
        assert_eq!(status, StatusCode::OK);
    }

    /// The chat route is guarded too, ahead of its body.
    #[tokio::test]
    async fn the_chat_route_is_guarded() {
        let mut state = AppState::headless("m", Arc::new(Config::default()));
        state.bind = addr("127.0.0.1:8080");
        let resp = build_router(state)
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("host", "evil.example:8080")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"m","messages":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }
}
