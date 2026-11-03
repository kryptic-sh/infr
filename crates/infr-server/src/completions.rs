//! `POST /v1/completions` — the legacy OpenAI text-completion endpoint: a raw prompt continued as
//! text, with no chat template. This is what FIM (`<|fim_prefix|>…`) clients and plain
//! autocompletion use.

use super::*;

#[derive(Debug, Deserialize)]
pub(crate) struct CompletionRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    prompt: serde_json::Value,
    #[serde(default)]
    suffix: Option<serde_json::Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    echo: bool,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<serde_json::Value>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
}

impl CompletionRequest {
    /// The sampling fields in the shape [`GenParams::from_request`] validates.
    fn sampling_shape(&self) -> ChatRequest {
        ChatRequest {
            model: self.model.clone(),
            messages: Vec::new(),
            stream: self.stream,
            tools: None,
            tool_choice: None,
            max_tokens: self.max_tokens,
            max_completion_tokens: None,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
            stop: self.stop.clone(),
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repeat_penalty: self.repeat_penalty,
            reasoning_effort: None,
            chat_template_kwargs: None,
            think: None,
        }
    }

    /// The single prompt string: a string, or a one-element array of one.
    fn prompt_text(&self) -> Result<String, ParamError> {
        let bad = |message: &str| ParamError {
            param: "prompt",
            message: message.into(),
        };
        match &self.prompt {
            serde_json::Value::String(s) => Ok(s.clone()),
            serde_json::Value::Array(a) if a.len() == 1 => a[0]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| bad("token-array prompts are not supported; send text")),
            serde_json::Value::Array(_) => Err(bad("batched prompts are not supported; send one")),
            serde_json::Value::Null => Err(bad("prompt is required")),
            _ => Err(bad("prompt must be a string")),
        }
    }

    fn validate(&self) -> Result<(), ParamError> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(ParamError {
                param: "n",
                message: "only n = 1 is supported".into(),
            });
        }
        if self.suffix.as_ref().is_some_and(|s| !s.is_null()) {
            return Err(ParamError {
                param: "suffix",
                message: "suffix is not supported; put the model's FIM tokens in the prompt".into(),
            });
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct TextCompletionResponse {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<TextChoice>,
    usage: UsageInfo,
}

#[derive(Serialize)]
struct TextChoice {
    text: String,
    index: usize,
    logprobs: Option<serde_json::Value>,
    finish_reason: Option<String>,
}

fn text_event(cid: &str, model: &str, created: i64, text: &str, finish: Option<&str>) -> Event {
    Event::default()
        .json_data(serde_json::json!({
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": null,
                "finish_reason": finish,
            }],
        }))
        .expect("a completion chunk always serializes")
}

pub(crate) async fn completions_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return param_error(None, e.body_text()),
    };
    let prompt = match req.validate().and_then(|()| req.prompt_text()) {
        Ok(p) => p,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    let mut params = match GenParams::from_request(&req.sampling_shape()) {
        Ok(p) => p,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    params.max_tokens = clamp_max_tokens(params.max_tokens, max_tokens_cap(&state.cfg));

    let entry = state.route(&req.model);
    let ctx = ReqCtx {
        id: next_req_id(),
        cid: make_id().replacen("chatcmpl-", "cmpl-", 1),
        model_id: entry.id.to_string(),
        created: unix_ts(),
        deadline: request_timeout(&state.cfg),
        stream: req.stream,
        stats: state.stats.clone(),
    };
    log_request_start(
        ctx.id,
        "/v1/completions",
        &ctx.model_id,
        1,
        prompt.len(),
        params.max_tokens,
        ctx.stream,
    );
    let echo = req.echo;
    if ctx.stream {
        streaming(entry, prompt, echo, params, ctx).await
    } else {
        non_streaming(entry, prompt, echo, params, ctx).await
    }
}

async fn non_streaming(
    entry: ModelEntry,
    prompt: String,
    echo: bool,
    params: GenParams,
    ctx: ReqCtx,
) -> Response {
    let ReqCtx {
        id: req_id,
        cid,
        model_id,
        created,
        deadline,
        stream,
        stats,
    } = ctx;
    let queued = QueuedGuard::new(stats.clone());
    let Ok(permit) = entry.slots.clone().acquire_owned().await else {
        stats.fold_failure();
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".into(),
        );
    };
    drop(queued);
    let active = ActiveGuard::new(stats.clone());
    let engine_arc = entry.engine.clone();
    let stats_blk = stats.clone();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_blk = cancel.clone();

    let mut handle = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let _permit = permit;
        let _active = active;
        let Some(engine) = engine_arc else {
            anyhow::bail!("no engine loaded");
        };
        let mut text = if echo { prompt.clone() } else { String::new() };
        let mut tally = ReqTally::new();
        let outcome = engine.complete(&prompt, &params, &cancel_blk, &mut |piece| {
            tally.on_text_delta(&stats_blk);
            text.push_str(piece);
        })?;
        Ok((text, outcome, tally))
    });

    let mut deadline_hit = false;
    let joined = match deadline {
        None => handle.await,
        Some(d) => match tokio::time::timeout(d, &mut handle).await {
            Ok(joined) => joined,
            Err(_) => {
                deadline_hit = true;
                cancel.store(true, Ordering::Relaxed);
                tracing::warn!(
                    req = req_id,
                    timeout_s = d.as_secs(),
                    "request deadline hit"
                );
                handle.await
            }
        },
    };

    match joined.map_err(anyhow::Error::from).and_then(|r| r) {
        Err(e) => {
            stats.fold_failure();
            tracing::warn!(req = req_id, error = %e, "request failed");
            json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        }
        Ok((text, outcome, tally)) => {
            let finish = if deadline_hit {
                Finish::Length
            } else {
                outcome.finish
            };
            let rec = tally.finish(outcome, finish);
            stats.fold_completion(&rec);
            log_request_done(req_id, &model_id, stream, &rec);
            Json(TextCompletionResponse {
                id: cid,
                object: "text_completion",
                created,
                model: model_id,
                choices: vec![TextChoice {
                    text,
                    index: 0,
                    logprobs: None,
                    finish_reason: Some(finish.as_str().into()),
                }],
                usage: UsageInfo {
                    prompt_tokens: outcome.prompt_tokens,
                    completion_tokens: outcome.completion_tokens,
                    total_tokens: outcome
                        .prompt_tokens
                        .saturating_add(outcome.completion_tokens),
                },
            })
            .into_response()
        }
    }
}

async fn streaming(
    entry: ModelEntry,
    prompt: String,
    echo: bool,
    params: GenParams,
    ctx: ReqCtx,
) -> Response {
    let ReqCtx {
        id: req_id,
        cid,
        model_id,
        created,
        deadline,
        stream,
        stats,
    } = ctx;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();

    let queued = QueuedGuard::new(stats.clone());
    let Ok(permit) = entry.slots.clone().acquire_owned().await else {
        stats.fold_failure();
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".into(),
        );
    };
    drop(queued);
    let active = ActiveGuard::new(stats.clone());
    let engine_arc = entry.engine.clone();
    let tx_cb = tx.clone();
    let cid_cb = cid.clone();
    let model_cb = model_id.clone();
    let stats_cb = stats.clone();

    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_cb = cancel.clone();
    let deadline_hit = Arc::new(AtomicBool::new(false));
    let deadline_hit_cb = deadline_hit.clone();
    let done_tx = deadline.map(|d| arm_deadline(d, cancel.clone(), deadline_hit.clone()));

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = active;
        let _done_tx = done_tx;
        let mut done = DoneGuard {
            tx: tx.clone(),
            req_id,
            stats: stats.clone(),
            settled: false,
        };
        let Some(engine) = engine_arc else {
            done.fail("no engine loaded");
            return;
        };
        if echo {
            let _ = tx.send(Ok(text_event(&cid, &model_id, created, &prompt, None)));
        }
        let mut tally = ReqTally::new();
        let res = engine.complete(&prompt, &params, &cancel_cb, &mut |piece| {
            tally.on_text_delta(&stats_cb);
            let ev = text_event(&cid_cb, &model_cb, created, piece, None);
            if tx_cb.send(Ok(ev)).is_err() {
                cancel_cb.store(true, Ordering::Relaxed);
            }
        });
        match res {
            Ok(outcome) => {
                let finish = if deadline_hit_cb.load(Ordering::Relaxed) {
                    Finish::Length
                } else {
                    outcome.finish
                };
                let _ = tx.send(Ok(text_event(
                    &cid,
                    &model_id,
                    created,
                    "",
                    Some(finish.as_str()),
                )));
                let rec = tally.finish(outcome, finish);
                stats.fold_completion(&rec);
                log_request_done(req_id, &model_id, stream, &rec);
                done.settled();
            }
            Err(e) => done.fail(&e.to_string()),
        }
    });

    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    Sse::new(stream).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// Echoes the raw prompt back in two pieces, so a test sees exactly what reached `complete`.
    struct RawGen;
    impl ChatGenerator for RawGen {
        fn status(&self) -> GeneratorStatus {
            GeneratorStatus {
                backend: "cpu",
                device: None,
                ctx_per_slot: None,
                memory: None,
            }
        }
        fn chat(
            &self,
            _: &[ChatMessage],
            _: Option<&serde_json::Value>,
            _: Option<&str>,
            _: &GenParams,
            _: &Arc<AtomicBool>,
            _: &mut dyn FnMut(Delta),
        ) -> anyhow::Result<ChatOutcome> {
            anyhow::bail!("chat not used")
        }
        fn complete(
            &self,
            prompt: &str,
            _: &GenParams,
            _: &Arc<AtomicBool>,
            on_text: &mut dyn FnMut(&str),
        ) -> anyhow::Result<ChatOutcome> {
            on_text("got:");
            on_text(prompt);
            Ok(ChatOutcome {
                finish: Finish::Length,
                prompt_tokens: 4,
                completion_tokens: 2,
            })
        }
    }

    fn router() -> Router {
        let g: Arc<dyn ChatGenerator> = Arc::new(RawGen);
        build_router(AppState::new(g, "m", 1, Arc::new(Config::default())))
    }

    async fn post(body: &str) -> (StatusCode, String) {
        let resp = router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_owned()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn returns_a_text_completion() {
        let (st, body) = post(r#"{"model":"m","prompt":"<|fim_prefix|>x"}"#).await;
        assert_eq!(st, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert!(v["id"].as_str().unwrap().starts_with("cmpl-"));
        assert_eq!(v["choices"][0]["text"], "got:<|fim_prefix|>x");
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(v["usage"]["total_tokens"], 6);
    }

    #[tokio::test]
    async fn echo_prepends_the_prompt_and_array_prompt_is_accepted() {
        let (_, body) = post(r#"{"prompt":["hi"],"echo":true}"#).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["choices"][0]["text"], "higot:hi");
    }

    #[tokio::test]
    async fn streams_chunks_then_finish_then_done() {
        let (st, body) = post(r#"{"prompt":"a","stream":true}"#).await;
        assert_eq!(st, StatusCode::OK);
        let frames: Vec<&str> = body
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .collect();
        assert_eq!(frames.last(), Some(&"[DONE]"));
        let first: serde_json::Value = serde_json::from_str(frames[0]).unwrap();
        assert_eq!(first["object"], "text_completion");
        assert_eq!(first["choices"][0]["text"], "got:");
        let fin: serde_json::Value = serde_json::from_str(frames[frames.len() - 2]).unwrap();
        assert_eq!(fin["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn bad_requests_are_openai_400s() {
        for (body, param) in [
            (r#"{"model":"m"}"#, "prompt"),
            (r#"{"prompt":["a","b"]}"#, "prompt"),
            (r#"{"prompt":[1,2]}"#, "prompt"),
            (r#"{"prompt":"a","n":2}"#, "n"),
            (r#"{"prompt":"a","suffix":"b"}"#, "suffix"),
            (r#"{"prompt":"a","temperature":9}"#, "temperature"),
        ] {
            let (st, out) = post(body).await;
            assert_eq!(st, StatusCode::BAD_REQUEST, "{body}");
            let v: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(v["error"]["param"], param, "{body}");
        }
    }

    #[tokio::test]
    async fn backend_without_complete_is_a_500() {
        let g: Arc<dyn ChatGenerator> = Arc::new(super::super::tests::EchoGen("x"));
        let r = build_router(AppState::new(g, "m", 1, Arc::new(Config::default())));
        let resp = r
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt":"a"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
