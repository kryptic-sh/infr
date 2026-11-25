//! `POST /v1/completions` — the legacy OpenAI text-completion endpoint: a raw prompt continued as
//! text, with no chat template. This is what FIM (`<|fim_prefix|>…`) clients and plain
//! autocompletion use.

use super::*;

/// Why a token-id prompt is refused. OpenAI's legacy route accepts `[1, 2, 3]`, this one takes
/// text only, because the raw path hands the backend a string.
const TOKEN_ARRAY: &str = "token-array prompts are not supported; send the prompt as text";

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
    /// llama.cpp/vLLM extension, as on the chat route (0 = off).
    #[serde(default)]
    min_p: Option<f32>,
    /// OpenAI `logit_bias`: `{"<token id>": bias}`. Kept as sent, for the same reason as on the
    /// chat route: a malformed map is a `400` naming the parameter, not a body-parse failure.
    #[serde(default)]
    logit_bias: Option<LogitBiasArg>,
    /// OpenAI `stream_options`; only `include_usage` is read.
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    /// OpenAI `best_of`: sample several server-side and return the best. Only the one choice `n`
    /// names is served, so this is refused unless it asks for the same.
    #[serde(default)]
    best_of: Option<u32>,
    /// OpenAI `user`: an end-user identifier the server does not use.
    #[serde(default)]
    user: Option<serde_json::Value>,
    /// OpenAI `logprobs` on the legacy route is a COUNT, where the chat route's is a flag. Both
    /// spellings are accepted: `true` with no `top_logprobs` means 20, and a number `n` means
    /// `top_logprobs: n` (evaluation harnesses send the number).
    #[serde(default)]
    logprobs: Option<LogprobsArg>,
    /// OpenAI `top_logprobs`: how many most likely alternatives to include per token.
    #[serde(default)]
    top_logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(untagged)]
enum LogprobsArg {
    Flag(bool),
    Count(u32),
}

impl CompletionRequest {
    /// The sampling fields in the shape [`GenParams::from_request`] validates.
    fn sampling_shape(&self) -> ChatRequest {
        // The legacy count spelling folds into the chat shape's flag plus a `top_logprobs` number.
        // An explicit `top_logprobs` wins; `0` means "this token only", so the flag goes off.
        let (logprobs, top_logprobs) = match self.logprobs {
            Some(LogprobsArg::Flag(b)) => (Some(b), self.top_logprobs.clone()),
            Some(LogprobsArg::Count(n)) => (
                Some(n > 0),
                self.top_logprobs
                    .clone()
                    .or_else(|| Some(serde_json::Value::from(n))),
            ),
            None => (None, self.top_logprobs.clone()),
        };
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
            min_p: self.min_p,
            logit_bias: self.logit_bias.clone(),
            n: None,
            logprobs,
            top_logprobs,
            stream_options: None,
            response_format: None,
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
            serde_json::Value::Array(a) if a.len() == 1 && a[0].is_string() => a[0]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| bad(TOKEN_ARRAY)),
            // Several strings: the batched shape OpenAI also accepts.
            serde_json::Value::Array(a) if a.iter().all(serde_json::Value::is_string) => {
                Err(bad("batched prompts are not supported; send one"))
            }
            serde_json::Value::Array(_) => Err(bad(TOKEN_ARRAY)),
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
        if self.best_of.is_some_and(|b| b != 1) {
            return Err(ParamError {
                param: "best_of",
                message: "only best_of = 1 is supported".into(),
            });
        }
        // Refused rather than dropped: a client that sets these is asking for something this
        // server will not do, and answering as if the field were absent hides that.
        if self.user.as_ref().is_some_and(|u| !u.is_null()) {
            return Err(ParamError {
                param: "user",
                message: "user is not supported".into(),
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
    logprobs: Option<ChatLogprobs>,
    finish_reason: Option<String>,
}

fn text_event(
    cid: &str,
    model: &str,
    created: i64,
    text: &str,
    logprobs: Option<ChatLogprobs>,
    finish: Option<&str>,
) -> Event {
    Event::default()
        .json_data(serde_json::json!({
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": logprobs,
                "finish_reason": finish,
            }],
        }))
        .expect("a completion chunk always serializes")
}

/// The `stream_options.include_usage` chunk for a text completion: the whole request's `usage`
/// with an empty `choices` array, sent after the finish chunk and before `[DONE]`.
fn text_usage_event(cid: &str, model: &str, created: i64, outcome: &ChatOutcome) -> Event {
    Event::default()
        .json_data(serde_json::json!({
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [],
            "usage": UsageInfo::from_outcome(outcome),
        }))
        .expect("a usage chunk always serializes")
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
        // A body the extractor refused keeps the status that says why (413 for an oversize body,
        // 415 for a missing JSON content type); everything else is the `400` an OpenAI client
        // expects, not axum's `422`.
        Err(e) => return body_rejection(e),
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
    // `logit_bias` ids depend on the model, so they are checked only once it is known — the same
    // check the chat route makes, and for the same reason: the ids are the model's vocabulary.
    if let (false, Some(engine)) = (params.logit_bias.is_empty(), &entry.engine) {
        if let Err(e) = params.check_vocab(engine.vocab_size()) {
            return param_error(Some(e.param), e.message);
        }
    }
    if params.logprobs {
        if let Some(engine) = &entry.engine {
            if !engine.supports_logprobs() {
                return param_error(
                    Some("logprobs"),
                    "logprobs: returning log probabilities is not supported by this backend".into(),
                );
            }
        }
    }
    let ctx = ReqCtx {
        id: next_req_id(),
        cid: make_id().replacen("chatcmpl-", "cmpl-", 1),
        model_id: entry.id.to_string(),
        created: unix_ts(),
        deadline: request_timeout(&state.cfg),
        keep_alive: sse_keep_alive(&state.cfg),
        stream: req.stream,
        include_usage: req.stream_options.as_ref().is_some_and(|o| o.include_usage),
        stats: state.stats.clone(),
    };

    // Count the prompt BEFORE queueing for a slot. An over-long one would otherwise wait for a
    // slot it can never use, and a streamed one would be handed a `200` before the overflow was
    // known. The chat path counts its rendered prompt for the same reason; a backend that cannot
    // count a raw prompt answers `None`, and its generator then reports the overflow, which
    // reaches the client as the same `400`.
    let mut prompt_tokens = None;
    let refused = match entry.engine.clone() {
        None => None,
        Some(engine) => {
            let probe = prompt.clone();
            match tokio::task::spawn_blocking(move || engine.prepare_completion(&probe)).await {
                Ok(Ok(counted)) => {
                    prompt_tokens = counted.map(|(n, _)| n);
                    match counted {
                        Some((tokens, Some(max))) if tokens > max => Some(
                            ContextLengthExceeded::prompt_too_long(
                                max.saturating_sub(1) as usize,
                                tokens as usize,
                            )
                            .into(),
                        ),
                        _ => None,
                    }
                }
                Ok(Err(e)) => Some(e),
                Err(e) => Some(anyhow::Error::from(e)),
            }
        }
    };

    log_request_start(
        ctx.id,
        RequestStart {
            route: "/v1/completions",
            model: &ctx.model_id,
            messages: 1,
            prompt_chars: prompt.len(),
            prompt_tokens,
            max_tokens: params.max_tokens,
            stream: ctx.stream,
        },
    );
    if let Some(e) = refused {
        ctx.stats.fold_failure();
        tracing::warn!(req = ctx.id, error = %e, "request refused before generation");
        return error_response(&e);
    }

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
        keep_alive: _,
        stream,
        include_usage: _,
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
    // A client that hangs up is noticed by hyper, which DROPS this handler future — and with it
    // this guard, which latches the abort so the generation stops and frees its slot instead of
    // running to `max_tokens` for nobody.
    let _hangup = CancelOnDrop(cancel.clone());
    let want_logprobs = params.logprobs;

    let mut handle = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let _permit = permit;
        let _active = active;
        let Some(engine) = engine_arc else {
            anyhow::bail!("no engine loaded");
        };
        let mut text = if echo { prompt.clone() } else { String::new() };
        let mut logprobs = Vec::new();
        let mut tally = ReqTally::new();
        let outcome = engine.complete_with_logprobs(
            &prompt,
            &params,
            &cancel_blk,
            &mut |piece| {
                tally.on_text_delta(&stats_blk);
                text.push_str(piece);
            },
            &mut |lp| logprobs.push(lp),
        )?;
        Ok((text, logprobs, outcome, tally))
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
            // The request's own fault (an over-long prompt, a refused sampling field) is a `400`
            // with the typed body the chat path sends, not a `500`.
            error_response(&e)
        }
        Ok((text, logprobs, outcome, tally)) => {
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
                    logprobs: want_logprobs.then(|| ChatLogprobs::from_tokens(&logprobs)),
                    finish_reason: Some(finish.as_str().into()),
                }],
                usage: UsageInfo::from_outcome(&outcome),
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
        keep_alive,
        stream,
        include_usage,
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
    // Notices a hang-up as it happens, during a long prefill — a failed `send` below only sees it
    // at the next delta.
    let hangup_done = cancel_on_hangup(&tx, cancel.clone());
    let deadline_hit = Arc::new(AtomicBool::new(false));
    let deadline_hit_cb = deadline_hit.clone();
    let done_tx = deadline.map(|d| arm_deadline(d, cancel.clone(), deadline_hit.clone()));

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = active;
        let _done_tx = done_tx;
        let _hangup_done = hangup_done;
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
            let _ = tx.send(Ok(text_event(
                &cid, &model_id, created, &prompt, None, None,
            )));
        }
        let mut tally = ReqTally::new();
        let pending_logprobs = std::cell::RefCell::new(Vec::new());
        let res = engine.complete_with_logprobs(
            &prompt,
            &params,
            &cancel_cb,
            &mut |piece| {
                tally.on_text_delta(&stats_cb);
                let mut pending = pending_logprobs.borrow_mut();
                let logprobs = if pending.is_empty() {
                    None
                } else {
                    Some(ChatLogprobs::from_tokens(&std::mem::take(&mut *pending)))
                };
                let ev = text_event(&cid_cb, &model_cb, created, piece, logprobs, None);
                if tx_cb.send(Ok(ev)).is_err() {
                    cancel_cb.store(true, Ordering::Relaxed);
                }
            },
            &mut |lp| pending_logprobs.borrow_mut().push(lp),
        );
        match res {
            Ok(outcome) => {
                let finish = if deadline_hit_cb.load(Ordering::Relaxed) {
                    Finish::Length
                } else {
                    outcome.finish
                };
                let pending_logprobs = pending_logprobs.into_inner();
                let last_logprobs = if pending_logprobs.is_empty() {
                    None
                } else {
                    Some(ChatLogprobs::from_tokens(&pending_logprobs))
                };
                let _ = tx.send(Ok(text_event(
                    &cid,
                    &model_id,
                    created,
                    "",
                    last_logprobs,
                    Some(finish.as_str()),
                )));
                if include_usage {
                    let _ = tx.send(Ok(text_usage_event(&cid, &model_id, created, &outcome)));
                }
                let rec = tally.finish(outcome, finish);
                stats.fold_completion(&rec);
                log_request_done(req_id, &model_id, stream, &rec);
                done.settled();
            }
            Err(e) => match request_error_body(&e) {
                Some(body) => done.fail_with(body, &e.to_string()),
                None => done.fail(&e.to_string()),
            },
        }
    });

    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    let sse = Sse::new(stream);
    match keep_alive {
        Some(d) => sse.keep_alive(KeepAlive::new().interval(d)).into_response(),
        None => sse.into_response(),
    }
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
        fn vocab_size(&self) -> usize {
            1000
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
                cached_tokens: 0,
            })
        }
    }

    /// A generator that reports one token's log probability and then its text.
    struct LogprobCompleteGen;
    impl ChatGenerator for LogprobCompleteGen {
        fn status(&self) -> GeneratorStatus {
            GeneratorStatus {
                backend: "cpu",
                device: None,
                ctx_per_slot: None,
                memory: None,
            }
        }
        fn vocab_size(&self) -> usize {
            1000
        }
        fn supports_logprobs(&self) -> bool {
            true
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
        fn complete_with_logprobs(
            &self,
            _prompt: &str,
            _params: &GenParams,
            _cancel: &Arc<AtomicBool>,
            on_text: &mut dyn FnMut(&str),
            on_logprob: &mut dyn FnMut(infr_engine::TokenLogprob),
        ) -> anyhow::Result<ChatOutcome> {
            on_logprob(infr_engine::TokenLogprob {
                token: 7,
                text: "hi".into(),
                bytes: b"hi".to_vec(),
                logprob: -0.25,
                top: vec![infr_engine::TokenLogprobAlt {
                    token: 9,
                    text: "ho".into(),
                    bytes: b"ho".to_vec(),
                    logprob: -1.5,
                }],
            });
            on_text("hi");
            Ok(ChatOutcome {
                finish: Finish::Stop,
                prompt_tokens: 1,
                completion_tokens: 1,
                cached_tokens: 0,
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

    /// The legacy route's `logprobs` is a COUNT, not the chat route's flag: an integer must be
    /// accepted as `top_logprobs`, not turned into a serde `400` with no field named.
    #[tokio::test]
    async fn the_legacy_integer_logprobs_spelling_is_accepted() {
        for body in [
            r#"{"model":"m","prompt":"a","logprobs":2}"#,
            r#"{"model":"m","prompt":"a","logprobs":2,"top_logprobs":1}"#,
        ] {
            let g: Arc<dyn ChatGenerator> = Arc::new(LogprobCompleteGen);
            let r = build_router(AppState::new(g, "m", 1, Arc::new(Config::default())));
            let resp = r
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
            assert_eq!(resp.status(), StatusCode::OK, "{body}");
            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                v["choices"][0]["logprobs"]["content"][0]["token"], "hi",
                "{body}"
            );
        }
    }

    #[tokio::test]
    async fn logprobs_are_returned_on_a_text_completion() {
        let g: Arc<dyn ChatGenerator> = Arc::new(LogprobCompleteGen);
        let r = build_router(AppState::new(g, "m", 1, Arc::new(Config::default())));
        let resp = r
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"m","prompt":"a","logprobs":true,"top_logprobs":1}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["text"], "hi");
        let lp = &v["choices"][0]["logprobs"]["content"][0];
        assert_eq!(lp["token"], "hi");
        assert_eq!(lp["logprob"], -0.25);
        assert_eq!(lp["top_logprobs"][0]["token"], "ho");
    }

    #[tokio::test]
    async fn streaming_completion_logprobs_ride_the_text_chunk() {
        let g: Arc<dyn ChatGenerator> = Arc::new(LogprobCompleteGen);
        let r = build_router(AppState::new(g, "m", 1, Arc::new(Config::default())));
        let resp = r
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"m","prompt":"a","stream":true,"logprobs":true,"top_logprobs":1}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        let chunk = body
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|f| *f != "[DONE]")
            .map(|f| serde_json::from_str::<serde_json::Value>(f).unwrap())
            .find(|v| v["choices"][0]["text"] == "hi")
            .expect("a text chunk");
        assert_eq!(chunk["choices"][0]["logprobs"]["content"][0]["token"], "hi");
    }

    #[tokio::test]
    async fn bad_requests_are_openai_400s() {
        for (body, param) in [
            (r#"{"model":"m"}"#, "prompt"),
            (r#"{"prompt":["a","b"]}"#, "prompt"),
            (r#"{"prompt":[1,2]}"#, "prompt"),
            (r#"{"prompt":"a","n":2}"#, "n"),
            (r#"{"prompt":"a","suffix":"b"}"#, "suffix"),
            (r#"{"prompt":"a","logprobs":true}"#, "logprobs"),
            (r#"{"prompt":"a","top_logprobs":2}"#, "top_logprobs"),
            (r#"{"prompt":"a","temperature":9}"#, "temperature"),
            (r#"{"prompt":"a","best_of":3}"#, "best_of"),
            (r#"{"prompt":"a","user":"u1"}"#, "user"),
            (r#"{"prompt":"a","logit_bias":{"1":-150}}"#, "logit_bias"),
            (r#"{"prompt":"a","logit_bias":{"2000":-1}}"#, "logit_bias"),
        ] {
            let (st, out) = post(body).await;
            assert_eq!(st, StatusCode::BAD_REQUEST, "{body}");
            let v: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(v["error"]["param"], param, "{body}");
        }
    }

    #[tokio::test]
    async fn stream_options_include_usage_appends_the_usage_chunk() {
        let (_, body) =
            post(r#"{"prompt":"a","stream":true,"stream_options":{"include_usage":true}}"#).await;
        let frames: Vec<&str> = body
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .collect();
        assert_eq!(frames.last(), Some(&"[DONE]"));
        let usage: serde_json::Value = serde_json::from_str(frames[frames.len() - 2]).unwrap();
        assert_eq!(usage["choices"], serde_json::json!([]));
        assert_eq!(usage["usage"]["total_tokens"], 6);
    }

    #[tokio::test]
    async fn backend_without_complete_is_a_400() {
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
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["param"], "model");
    }
}
