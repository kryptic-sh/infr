use super::*;
use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;

struct BlockingGen {
    entered: tokio::sync::mpsc::UnboundedSender<()>,
    release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
    preflight: bool,
}

impl BlockingGen {
    fn wait(&self) -> anyhow::Result<()> {
        self.entered.send(())?;
        self.release
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))?;
        Ok(())
    }
}

impl ChatGenerator for BlockingGen {
    fn status(&self) -> GeneratorStatus {
        GeneratorStatus {
            backend: "cpu",
            device: None,
            ctx_per_slot: None,
            memory: None,
        }
    }

    fn validate_images(&self, _images: &[Arc<[u8]>]) -> anyhow::Result<()> {
        if self.preflight {
            self.wait()?;
        }
        Ok(())
    }

    fn chat(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&serde_json::Value>,
        _tool_choice: Option<&str>,
        _params: &GenParams,
        _cancel: &Arc<AtomicBool>,
        on_delta: &mut dyn FnMut(Delta),
    ) -> anyhow::Result<ChatOutcome> {
        if !self.preflight {
            self.wait()?;
        }
        on_delta(Delta::Content("done".into()));
        Ok(ChatOutcome {
            finish: Finish::Stop,
            prompt_tokens: 1,
            completion_tokens: 1,
        })
    }

    fn complete(
        &self,
        _prompt: &str,
        _params: &GenParams,
        _cancel: &Arc<AtomicBool>,
        on_text: &mut dyn FnMut(&str),
    ) -> anyhow::Result<ChatOutcome> {
        self.wait()?;
        on_text("done");
        Ok(ChatOutcome {
            finish: Finish::Stop,
            prompt_tokens: 1,
            completion_tokens: 1,
        })
    }
}

fn request(raw: bool, stream: bool, image: bool) -> Request<Body> {
    let body = if raw {
        serde_json::json!({"model":"m", "prompt":"hello", "stream":stream})
    } else {
        let content = if image {
            serde_json::json!([{"type":"image_url", "image_url":{"url":"data:image/png;base64,AQID"}}])
        } else {
            serde_json::json!("hello")
        };
        serde_json::json!({"model":"m", "messages":[{"role":"user", "content":content}], "stream":stream})
    };
    Request::builder()
        .method("POST")
        .uri(if raw {
            "/v1/completions"
        } else {
            "/v1/chat/completions"
        })
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test(start_paused = true)]
async fn queued_requests_keep_server_busy() {
    for raw in [false, true] {
        for stream in [false, true] {
            let state = AppState::new(
                Arc::new(tests::EchoGen("m")),
                "m",
                1,
                Arc::new(Config::default()),
            );
            let permit = state.models[0].slots.clone().acquire_owned().await.unwrap();
            let activity = state.activity.clone();
            let task = tokio::spawn(build_router(state).oneshot(request(raw, stream, false)));
            tokio::task::yield_now().await;
            assert_eq!(activity.busy(), 1);
            let waiter = tokio::spawn({
                let activity = activity.clone();
                async move { activity.wait_idle(Duration::from_secs(5)).await }
            });
            tokio::time::advance(Duration::from_secs(50)).await;
            assert!(!waiter.is_finished());
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            assert_eq!(activity.busy(), 0);
            drop(permit);
            tokio::time::advance(Duration::from_secs(4)).await;
            assert!(!waiter.is_finished());
            tokio::time::advance(Duration::from_secs(1)).await;
            waiter.await.unwrap();
        }
    }
}

#[tokio::test(start_paused = true)]
async fn disconnected_generation_and_preflight_keep_server_busy() {
    for (raw, stream, image) in [
        (false, false, false),
        (false, true, false),
        (true, false, false),
        (true, true, false),
        (false, false, true),
    ] {
        let (entered, mut started) = tokio::sync::mpsc::unbounded_channel();
        let (release, receiver) = std::sync::mpsc::channel();
        let state = AppState::new(
            Arc::new(BlockingGen {
                entered,
                release: std::sync::Mutex::new(receiver),
                preflight: image,
            }),
            "m",
            1,
            Arc::new(Config::default()),
        );
        let slots = state.models[0].slots.clone();
        let activity = state.activity.clone();
        let task = tokio::spawn(build_router(state).oneshot(request(raw, stream, image)));
        started.recv().await.unwrap();
        if stream {
            let response = task.await.unwrap().unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            drop(response);
        } else {
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
        }
        assert_eq!(
            activity.busy(),
            1,
            "raw={raw} stream={stream} image={image}"
        );
        let waiter = tokio::spawn({
            let activity = activity.clone();
            async move { activity.wait_idle(Duration::from_secs(5)).await }
        });
        tokio::time::advance(Duration::from_secs(50)).await;
        assert!(!waiter.is_finished());
        release.send(()).unwrap();
        if !image {
            let permit = slots.acquire_owned().await.unwrap();
            drop(permit);
        }
        for _ in 0..1000 {
            if activity.busy() == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(activity.busy(), 0);
        tokio::time::advance(Duration::from_secs(4)).await;
        assert!(!waiter.is_finished());
        tokio::time::advance(Duration::from_secs(1)).await;
        waiter.await.unwrap();
    }
}

#[tokio::test(start_paused = true)]
async fn probes_and_errors_reset_idle_and_stopped_admission_returns_503() {
    let state = AppState::headless("m", Arc::new(Config::default()));
    let activity = state.activity.clone();
    let router = build_router(state);
    let waiter = tokio::spawn({
        let activity = activity.clone();
        async move { activity.wait_idle(Duration::from_secs(5)).await }
    });
    for path in ["/health", "/missing", "/metrics", "/status"] {
        tokio::time::advance(Duration::from_secs(4)).await;
        let response = router
            .clone()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        response.into_body().collect().await.unwrap();
        assert!(!waiter.is_finished());
    }
    tokio::time::advance(Duration::from_secs(5)).await;
    waiter.await.unwrap();
    let response = router
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(!infr_core::shutdown::shutdown_requested());
}

#[tokio::test(start_paused = true)]
async fn zero_idle_timeout_is_disabled() {
    let task = tokio::spawn(server_shutdown(Arc::new(idle::Activity::default()), 0));
    tokio::time::advance(Duration::from_secs(86400)).await;
    assert!(!task.is_finished());
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
}

#[tokio::test]
async fn listener_exits_on_idle_and_closes_unused_keepalive() {
    let mut cfg = Config::default();
    cfg.serve.idle_timeout_secs = 1;
    let state = AppState::headless("m", Arc::new(cfg));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_listener(state, listener));
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();
    let mut response = [0; 1024];
    let n = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut response))
        .await
        .unwrap()
        .unwrap();
    assert!(std::str::from_utf8(&response[..n])
        .unwrap()
        .starts_with("HTTP/1.1 200"));
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), stream.read(&mut response))
            .await
            .unwrap()
            .unwrap(),
        0
    );
    assert!(tokio::net::TcpStream::connect(addr).await.is_err());
}

#[tokio::test]
async fn socket_activation_child() {
    if std::env::var("LISTEN_PID") != Ok(std::process::id().to_string()) {
        return;
    }
    let mut cfg = Config::default();
    cfg.serve.idle_timeout_secs = 1;
    serve_state(
        AppState::headless("m", Arc::new(cfg)),
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap();
}

struct ChildGuard(std::process::Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn systemd_socket_activation_serves_and_exits() {
    let reserve = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = reserve.local_addr().unwrap();
    drop(reserve);
    let child = std::process::Command::new("systemd-socket-activate")
        .arg(format!("--listen={addr}"))
        .arg(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "idle_tests::socket_activation_child",
            "--nocapture",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    let mut child = match child {
        Ok(child) => ChildGuard(child),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
        Err(error) => panic!("could not start systemd-socket-activate: {error}"),
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut stream = loop {
        if let Ok(stream) = tokio::net::TcpStream::connect(addr).await {
            break stream;
        }
        assert!(
            Instant::now() < deadline,
            "socket activation never opened its listener"
        );
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "socket launcher exited early"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(8), stream.read_to_end(&mut response))
        .await
        .unwrap()
        .unwrap();
    assert!(std::str::from_utf8(&response)
        .unwrap()
        .starts_with("HTTP/1.1 200"));
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(
                status.success(),
                "activated server did not exit cleanly: {status}"
            );
            break;
        }
        assert!(
            Instant::now() < deadline,
            "activated server did not exit on idle"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
