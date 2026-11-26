//! `POST /v1/completions` — the legacy OpenAI text-completion endpoint: a raw prompt continued as
//! text, with no chat template. This is what FIM (`<|fim_prefix|>…`) clients and plain
//! autocompletion use.

use super::*;

/// Why a token-id prompt is refused. OpenAI's legacy route accepts `[1, 2, 3]`, this one takes
/// text only, because the raw path hands the backend a string.
const TOKEN_ARRAY: &str = "token-array prompts are not supported; send the prompt as text";

/// The request field a raw completion's prompt comes from, which its prompt errors name.
const INPUT: &str = "prompt";

/// A `/v1/completions` request (OpenAI's legacy wire format). Unknown fields are ignored, as on the
/// chat route — `user` among them, which OpenAI clients send routinely.
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
    /// `-1` means "not set", as on the chat route: the server's default reply budget.
    #[serde(default, deserialize_with = "minus_one_is_unset")]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i64>,
    /// `-1` means "not set" (a random seed), as on the chat route.
    #[serde(default, deserialize_with = "minus_one_is_unset")]
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
    /// OpenAI `logprobs` on the legacy route is a COUNT, where the chat route's is a flag. Both
    /// spellings are accepted: `true` with no `top_logprobs` means 20, and a number `n` means
    /// `top_logprobs: n` (evaluation harnesses send the number). `0` still asks for the chosen
    /// tokens' log probabilities, with no alternatives.
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
        // An explicit `top_logprobs` wins. Any count turns the flag on: `0` is "the chosen token
        // only", not "off" (that is an absent or null `logprobs`).
        let (logprobs, top_logprobs) = match self.logprobs {
            Some(LogprobsArg::Flag(b)) => (Some(b), self.top_logprobs.clone()),
            Some(LogprobsArg::Count(n)) => (
                Some(true),
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
        // The count becomes `top_logprobs` only when that is absent; checked here so the error
        // names the field the client sent.
        if let (Some(LogprobsArg::Count(n)), None) = (self.logprobs, &self.top_logprobs) {
            if n as usize > MAX_TOP_LOGPROBS {
                return Err(ParamError {
                    param: "logprobs",
                    message: format!("logprobs must be at most {MAX_TOP_LOGPROBS}, got {n}"),
                });
            }
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

    // Count the prompt BEFORE queueing for a slot, under the same bound the chat route prepares
    // under. An over-long one would otherwise wait for a slot it can never use, and a streamed one
    // would be handed a `200` before the overflow was known. A backend that cannot count a raw
    // prompt answers `None`, and its generator then reports the overflow, which reaches the client
    // as the same `400`.
    let (prompt, counted) = match entry.engine.clone() {
        None => (prompt, Ok(None)),
        Some(engine) => {
            let joined = prepare_bounded(&entry, &ctx, move || {
                let counted = engine.prepare_completion(&prompt);
                (prompt, counted)
            })
            .await;
            match joined {
                Ok(parts) => parts,
                Err(resp) => return *resp,
            }
        }
    };
    let prompt_tokens = counted.as_ref().ok().copied().flatten().map(|(n, _)| n);
    let refused = match counted {
        Err(e) => Some(e),
        // Nothing to continue from. Refused only when the count is known: without one, a model
        // that adds a BOS still has a token to continue from, and the generator reports an empty
        // prompt as this same error.
        Ok(Some((0, _))) => Some(ParamError::empty_prompt().into()),
        Ok(Some((tokens, Some(max)))) if tokens > max => Some(
            ContextLengthExceeded::prompt_too_long(INPUT, max as usize, tokens as usize).into(),
        ),
        Ok(_) => None,
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
        return error_response(&e, INPUT);
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
    let stats_blk = ctx.stats.clone();
    let want_logprobs = params.logprobs;
    let ran = run_unary(&entry, &ctx, INPUT, move |engine, cancel| {
        let mut text = if echo { prompt.clone() } else { String::new() };
        let mut logprobs = Vec::new();
        let mut tally = ReqTally::new();
        let outcome = engine.complete_with_logprobs(
            &prompt,
            &params,
            cancel,
            &mut |piece| {
                tally.on_text_delta(&stats_blk);
                text.push_str(piece);
            },
            &mut |lp| logprobs.push(lp),
        )?;
        Ok((text, logprobs, outcome, tally))
    })
    .await;
    let ((text, logprobs, outcome, tally), deadline_hit) = match ran {
        Ok(ran) => ran,
        Err(resp) => return *resp,
    };
    let ReqCtx {
        id: req_id,
        cid,
        model_id,
        created,
        stream,
        stats,
        ..
    } = ctx;
    // A deadline hit is a truncation, as on the chat route: the generator saw only an abort.
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

async fn streaming(
    entry: ModelEntry,
    prompt: String,
    echo: bool,
    params: GenParams,
    ctx: ReqCtx,
) -> Response {
    // The job's own copy: it runs on the blocking thread, past this borrow of `ctx`.
    let ReqCtx {
        id: req_id,
        cid,
        model_id,
        created,
        stream,
        include_usage,
        stats,
        ..
    } = ctx.clone();
    run_streamed(&entry, &ctx, move |mut job| {
        let Some(engine) = job.engine.clone() else {
            job.done.fail("no engine loaded");
            return;
        };
        if echo {
            job.send(text_event(&cid, &model_id, created, &prompt, None, None));
        }
        let mut tally = ReqTally::new();
        let pending_logprobs = std::cell::RefCell::new(Vec::new());
        let res = engine.complete_with_logprobs(
            &prompt,
            &params,
            &job.cancel,
            &mut |piece| {
                tally.on_text_delta(&stats);
                let mut pending = pending_logprobs.borrow_mut();
                let logprobs = if pending.is_empty() {
                    None
                } else {
                    Some(ChatLogprobs::from_tokens(&std::mem::take(&mut *pending)))
                };
                job.send(text_event(&cid, &model_id, created, piece, logprobs, None));
            },
            &mut |lp| pending_logprobs.borrow_mut().push(lp),
        );
        match res {
            Ok(outcome) => {
                let finish = job.finish(outcome.finish);
                let pending_logprobs = pending_logprobs.into_inner();
                let last_logprobs = if pending_logprobs.is_empty() {
                    None
                } else {
                    Some(ChatLogprobs::from_tokens(&pending_logprobs))
                };
                job.send(text_event(
                    &cid,
                    &model_id,
                    created,
                    "",
                    last_logprobs,
                    Some(finish.as_str()),
                ));
                if include_usage {
                    job.send(text_usage_event(&cid, &model_id, created, &outcome));
                }
                let rec = tally.finish(outcome, finish);
                stats.fold_completion(&rec);
                log_request_done(req_id, &model_id, stream, &rec);
                job.done.settled();
            }
            Err(e) => job.fail(&e, INPUT),
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// The prompt budget [`RawGen`] counts against, in its tokens (one per byte).
    const RAW_LIMIT: u32 = 64;

    /// Echoes the raw prompt back in two pieces, so a test sees exactly what reached `complete`,
    /// and keeps the params it was handed. It counts one token per prompt byte against
    /// [`RAW_LIMIT`], and refuses an empty prompt the way the engine does — with an error that is
    /// not the request's typed fault.
    #[derive(Default)]
    struct RawGen {
        seen: std::sync::Mutex<Option<GenParams>>,
    }
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
        fn prepare_completion(&self, prompt: &str) -> anyhow::Result<Option<(u32, Option<u32>)>> {
            Ok(Some((prompt.len() as u32, Some(RAW_LIMIT))))
        }
        fn complete(
            &self,
            prompt: &str,
            params: &GenParams,
            _: &Arc<AtomicBool>,
            on_text: &mut dyn FnMut(&str),
        ) -> anyhow::Result<ChatOutcome> {
            anyhow::ensure!(!prompt.is_empty(), "empty prompt: nothing to generate from");
            *self.seen.lock().unwrap() = Some(params.clone());
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
            params: &GenParams,
            _cancel: &Arc<AtomicBool>,
            on_text: &mut dyn FnMut(&str),
            on_logprob: &mut dyn FnMut(infr_engine::TokenLogprob),
        ) -> anyhow::Result<ChatOutcome> {
            assert!(params.logprobs, "the server only asks when requested");
            let mut top = vec![infr_engine::TokenLogprobAlt {
                token: 9,
                text: "ho".into(),
                bytes: b"ho".to_vec(),
                logprob: -1.5,
            }];
            top.truncate(params.top_logprobs);
            on_logprob(infr_engine::TokenLogprob {
                token: 7,
                text: "hi".into(),
                bytes: b"hi".to_vec(),
                logprob: -0.25,
                top,
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
        router_for(Arc::new(RawGen::default()))
    }

    fn router_for(g: Arc<dyn ChatGenerator>) -> Router {
        build_router(AppState::new(g, "m", 1, Arc::new(Config::default())))
    }

    async fn post(body: &str) -> (StatusCode, String) {
        post_to(router(), body).await
    }

    async fn post_to(router: Router, body: &str) -> (StatusCode, String) {
        let resp = router
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

    /// The legacy `logprobs: 0` asks for the chosen tokens' log probabilities with no
    /// alternatives: entries with an empty `top_logprobs`, not "off" and not a `400`.
    #[tokio::test]
    async fn a_zero_legacy_logprobs_count_returns_entries_without_alternatives() {
        for body in [
            r#"{"model":"m","prompt":"a","logprobs":0}"#,
            r#"{"model":"m","prompt":"a","logprobs":0,"stream":true}"#,
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
            let body_text = String::from_utf8(bytes.to_vec()).unwrap();
            let v: serde_json::Value = match body_text.strip_prefix("data: ") {
                Some(_) => body_text
                    .lines()
                    .filter_map(|l| l.strip_prefix("data: "))
                    .filter(|f| *f != "[DONE]")
                    .map(|f| serde_json::from_str::<serde_json::Value>(f).unwrap())
                    .find(|v| v["choices"][0]["text"] == "hi")
                    .expect("a text chunk"),
                None => serde_json::from_str(&body_text).unwrap(),
            };
            let lp = &v["choices"][0]["logprobs"]["content"][0];
            assert_eq!(lp["token"], "hi", "{body}");
            assert_eq!(lp["logprob"], -0.25, "{body}");
            assert_eq!(lp["top_logprobs"], serde_json::json!([]), "{body}");
        }
    }

    /// `echo` returns the prompt ahead of the reply, but the entries are the generated tokens'
    /// only: they spell the text after the prompt.
    #[tokio::test]
    async fn echoed_prompt_text_carries_no_entries() {
        let g: Arc<dyn ChatGenerator> = Arc::new(LogprobCompleteGen);
        let r = build_router(AppState::new(g, "m", 1, Arc::new(Config::default())));
        let resp = r
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"m","prompt":"say ","echo":true,"logprobs":1}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let text = v["choices"][0]["text"].as_str().unwrap();
        let spelled: String = v["choices"][0]["logprobs"]["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["token"].as_str().unwrap())
            .collect();
        assert_eq!(text, "say hi");
        assert_eq!(spelled, "hi");
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
            (r#"{"prompt":"a","logprobs":21}"#, "logprobs"),
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

    /// An empty prompt whose count is known to be zero is the request's fault, refused before it
    /// queues: a `400` naming `prompt`, on a stream too.
    #[tokio::test]
    async fn an_empty_prompt_is_a_400_naming_prompt() {
        for body in [r#"{"prompt":""}"#, r#"{"prompt":[""],"stream":true}"#] {
            let (st, out) = post(body).await;
            assert_eq!(st, StatusCode::BAD_REQUEST, "{body}: {out}");
            let v: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(v["error"]["type"], "invalid_request_error", "{body}");
            assert_eq!(v["error"]["param"], "prompt", "{body}");
        }
    }

    /// Cannot count a raw prompt. With `bos` it adds a token to every prompt, so an empty one
    /// generates; without, it refuses an empty prompt as [`ChatGenerator::complete`] asks.
    struct UncountedGen {
        bos: bool,
    }
    impl ChatGenerator for UncountedGen {
        fn status(&self) -> GeneratorStatus {
            RawGen::default().status()
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
            if prompt.is_empty() && !self.bos {
                return Err(ParamError::empty_prompt().into());
            }
            on_text("ok");
            Ok(ChatOutcome {
                finish: Finish::Stop,
                prompt_tokens: 1,
                completion_tokens: 1,
                cached_tokens: 0,
            })
        }
    }

    /// Without a count the server does not guess: an empty prompt goes on to what any prompt would
    /// get — a model that adds a BOS generates from it, the generator's typed refusal is a `400`
    /// naming `prompt` (an error frame once a stream's `200` is out, as on the chat route), and no
    /// model loaded is the same error as for any other prompt.
    #[tokio::test]
    async fn without_a_count_an_empty_prompt_gets_what_any_prompt_would() {
        let (st, out) = post_to(
            router_for(Arc::new(UncountedGen { bos: true })),
            r#"{"prompt":""}"#,
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{out}");

        let no_bos = || router_for(Arc::new(UncountedGen { bos: false }));
        let (st, out) = post_to(no_bos(), r#"{"prompt":""}"#).await;
        assert_eq!(st, StatusCode::BAD_REQUEST, "{out}");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"]["param"], "prompt");
        let (st, out) = post_to(no_bos(), r#"{"prompt":"","stream":true}"#).await;
        assert_eq!(st, StatusCode::OK);
        let frame = out
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .find(|f| f.contains("\"error\""))
            .expect("an error frame");
        let v: serde_json::Value = serde_json::from_str(frame).unwrap();
        assert_eq!(v["error"]["param"], "prompt");

        let headless = || build_router(AppState::headless("m", Arc::new(Config::default())));
        let (empty, _) = post_to(headless(), r#"{"prompt":""}"#).await;
        let (text, _) = post_to(headless(), r#"{"prompt":"a"}"#).await;
        assert_eq!(empty, text);
        assert_eq!(empty, StatusCode::INTERNAL_SERVER_ERROR);
    }

    /// A prompt past the model's budget names `prompt`, the budget and the prompt's own count —
    /// on a stream too, where it must still be a `400` and not a `200` with an error frame.
    #[tokio::test]
    async fn an_over_long_prompt_names_prompt_and_both_counts() {
        let long = "x".repeat(RAW_LIMIT as usize + 1);
        for stream in [false, true] {
            let body = format!(r#"{{"prompt":"{long}","stream":{stream}}}"#);
            let (st, out) = post(&body).await;
            assert_eq!(st, StatusCode::BAD_REQUEST, "stream={stream}: {out}");
            let v: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(v["error"]["param"], "prompt");
            assert_eq!(v["error"]["code"], "context_length_exceeded");
            let msg = v["error"]["message"].as_str().unwrap();
            assert!(
                msg.contains(&format!("maximum context length is {RAW_LIMIT} tokens")),
                "{msg}"
            );
            assert!(
                msg.contains(&format!("your prompt resulted in {} tokens", RAW_LIMIT + 1)),
                "{msg}"
            );
        }
        // At the budget exactly it fits.
        let fits = format!(r#"{{"prompt":"{}"}}"#, "x".repeat(RAW_LIMIT as usize));
        assert_eq!(post(&fits).await.0, StatusCode::OK);
    }

    /// Cannot count a raw prompt, so the overflow comes from generation itself.
    struct OverflowGen;
    impl ChatGenerator for OverflowGen {
        fn status(&self) -> GeneratorStatus {
            RawGen::default().status()
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
            _: &str,
            _: &GenParams,
            _: &Arc<AtomicBool>,
            _: &mut dyn FnMut(&str),
        ) -> anyhow::Result<ChatOutcome> {
            Err(ContextLengthExceeded::prompt_too_long(INPUT, 8, 9).into())
        }
    }

    /// An overflow the generator reports names `prompt` too: a `400` when not streaming, the
    /// typed error frame when streaming.
    #[tokio::test]
    async fn a_generator_reported_overflow_names_prompt() {
        let (st, out) = post_to(router_for(Arc::new(OverflowGen)), r#"{"prompt":"a"}"#).await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"]["param"], "prompt");
        assert_eq!(v["error"]["code"], "context_length_exceeded");

        let (st, out) = post_to(
            router_for(Arc::new(OverflowGen)),
            r#"{"prompt":"a","stream":true}"#,
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        let frame = out
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .find(|f| f.contains("\"error\""))
            .expect("an error frame");
        let v: serde_json::Value = serde_json::from_str(frame).unwrap();
        assert_eq!(v["error"]["param"], "prompt");
        assert_eq!(v["error"]["code"], "context_length_exceeded");
    }

    /// `-1` for `max_tokens` and `seed` means unset, as on the chat route; any other negative is
    /// still refused.
    #[tokio::test]
    async fn minus_one_max_tokens_and_seed_mean_unset() {
        let g = Arc::new(RawGen::default());
        let (st, out) = post_to(
            router_for(g.clone()),
            r#"{"prompt":"a","max_tokens":-1,"seed":-1}"#,
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{out}");
        let seen = g.seen.lock().unwrap().clone().expect("complete ran");
        assert_eq!((seen.max_tokens, seen.seed), (None, None));
        let (st, _) = post_to(
            router_for(g.clone()),
            r#"{"prompt":"a","max_tokens":7,"seed":3}"#,
        )
        .await;
        assert_eq!(st, StatusCode::OK);
        let seen = g.seen.lock().unwrap().clone().expect("complete ran");
        assert_eq!((seen.max_tokens, seen.seed), (Some(7), Some(3)));
        for bad in [r#""max_tokens":-2"#, r#""seed":-5"#] {
            let (st, _) = post(&format!(r#"{{"prompt":"a",{bad}}}"#)).await;
            assert_eq!(st, StatusCode::BAD_REQUEST, "{bad}");
        }
    }

    /// OpenAI clients send `user` routinely; it is accepted and ignored, as on the chat route.
    #[tokio::test]
    async fn user_is_accepted_and_ignored() {
        let (st, out) = post(r#"{"prompt":"a","user":"u1"}"#).await;
        assert_eq!(st, StatusCode::OK, "{out}");
    }

    /// A token-id prompt is refused as one, not as a batch.
    #[tokio::test]
    async fn a_token_array_prompt_says_token_arrays_are_unsupported() {
        for body in [r#"{"prompt":[1,2]}"#, r#"{"prompt":[[1,2]]}"#] {
            let (st, out) = post(body).await;
            assert_eq!(st, StatusCode::BAD_REQUEST, "{body}");
            let v: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(v["error"]["message"], TOKEN_ARRAY, "{body}");
        }
    }

    /// A body the extractor refuses keeps the status that says why.
    #[tokio::test]
    async fn a_body_without_a_json_content_type_keeps_its_415() {
        let resp = router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .body(Body::from(r#"{"prompt":"a"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }
}
