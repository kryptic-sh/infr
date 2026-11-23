//! Serving the router over a Unix-domain socket: the `axum::serve` path an inherited
//! `ListenStream=%t/infr.sock` takes.

use std::{path::Path, sync::Arc, time::Duration};

use infr_core::config::Config;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::{host::Bind, serve_listener, AppState};

/// `GET uri` over a fresh connection to the Unix socket, with `Host: host`, and the whole
/// response. `Connection: close` ends the read.
async fn get(path: &Path, uri: &str, host: &str) -> String {
    let mut stream = tokio::net::UnixStream::connect(path).await.unwrap();
    let request = format!("GET {uri} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut response = String::new();
    tokio::time::timeout(Duration::from_secs(8), stream.read_to_string(&mut response))
        .await
        .expect("response timed out")
        .unwrap();
    response
}

fn state(cfg: Config) -> AppState {
    let mut state = AppState::headless("m", Arc::new(cfg));
    state.bind = Some(Bind::Unix);
    state
}

/// The socket accepts real HTTP and the default `Host` policy does not apply to it: a name a
/// browser would never send to a Unix socket is answered too.
#[tokio::test]
async fn serves_http_over_a_unix_socket() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("infr.sock");
    let listener = tokio::net::UnixListener::bind(&path).unwrap();
    let server = tokio::spawn(serve_listener(state(Config::default()), listener));

    let response = get(&path, "/v1/models", "evil.example").await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("\"object\":\"list\""), "{response}");
    server.abort();
}

/// An explicit `serve.allowed_hosts` turns the check on for a Unix socket too.
#[tokio::test]
async fn an_allow_list_still_gates_a_unix_socket() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("infr.sock");
    let listener = tokio::net::UnixListener::bind(&path).unwrap();
    let mut cfg = Config::default();
    cfg.serve.allowed_hosts = Some(vec!["llm.example.com".to_owned()]);
    let server = tokio::spawn(serve_listener(state(cfg), listener));

    let response = get(&path, "/v1/models", "llm.example.com").await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let response = get(&path, "/v1/models", "evil.example").await;
    assert!(response.starts_with("HTTP/1.1 403"), "{response}");
    server.abort();
}
