//! OpenAI-compatible HTTP server (axum + SSE). Talks only to `infr-engine` — never the GPU.
//!
//! Reference for the wire mapping (streaming, `reasoning_content`, tool_calls): the working
//! shim at `~/Projects/scratch/dgemma-openai-server.py`. See docs/PLAN.md "server".
//!
//! Routes:
//!   GET  /health                -> 200 OK
//!   GET  /v1/models             -> { object: "list", data: [{ id, object, owned_by }] }
//!   POST /v1/chat/completions   -> chat.completion | SSE chat.completion.chunk stream
//!
//! Delta mapping:
//!   `Delta::Reasoning`  -> `delta.reasoning_content`
//!   `Delta::Content`    -> `delta.content`
//!   `Delta::ToolCall`   -> `delta.tool_calls[]`  (finish_reason "tool_calls")

use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use infr_engine::{ChatMessage, Delta, ToolCall};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

/// Why generation ended — the OpenAI `finish_reason`. The generator reports it; the handlers
/// serialize it (a tool call still overrides to [`Finish::ToolCalls`] at the wire layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// EOS, or a `stop` sequence fired.
    Stop,
    /// The `max_tokens` / `max_completion_tokens` budget was exhausted.
    Length,
    /// A tool call was emitted.
    ToolCalls,
}

impl Finish {
    pub fn as_str(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::ToolCalls => "tool_calls",
        }
    }
}

/// Generation backend the server drives — it never knows the model/GPU underneath. Implemented by
/// the CLI's per-arch adapters (`infr-cli`'s `ParallelGenerator` over `infr_llama::ParallelSeam` for
/// the Vulkan seam; `SeamGenerator` wraps any `infr_llama::ChatModel` for the rest, including
/// `DiffusionGemmaChat` — see `docs/DIFFUSIONGEMMA.md`).
///
/// [`GenParams`] carries the request's PER-REQUEST sampling config (temperature/top_p/top_k/seed/
/// penalties/stop/max_tokens). Every field is an `Option` whose `None` means "inherit the process
/// default" — so a request that sends nothing generates EXACTLY as it did before this existed.
///
/// **`&self`, `Send + Sync`.** This is the whole concurrency contract: the server calls `chat` from
/// N request tasks at once and the generator is responsible for its own slot allocation and GPU
/// turn-taking. It used to be `&mut self` behind an `Arc<Mutex<_>>` the handlers held for an ENTIRE
/// generation, which meant request #2 waited for request #1 to finish — head-of-line blocking, no
/// parallelism. An implementation that genuinely cannot run concurrently (CPU / Metal / diffusion
/// today) keeps an internal `Mutex` and is served with `--parallel 1`, which is honest rather than
/// silently serialising a server the user asked to parallelise.
pub trait ChatGenerator: Send + Sync {
    fn chat(
        &self,
        messages: &[ChatMessage],
        tools_json: Option<&str>,
        tool_choice: Option<&str>,
        params: &GenParams,
        on_delta: &mut dyn FnMut(Delta),
    ) -> anyhow::Result<Finish>;
}

// ---------------------------------------------------------------------------
// Request DTOs
// ---------------------------------------------------------------------------

/// Top-level chat completion request (OpenAI wire format).
///
/// Unknown fields are IGNORED (no `deny_unknown_fields`) — an OpenAI client sending `n`, `user`,
/// `logit_bias`, … must not 400. Known-but-invalid VALUES do 400, via [`GenParams::from_request`].
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessageDto>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    /// OpenAI `tool_choice`: `"auto"` | `"required"` | `"none"` | `{"type":"function","function":
    /// {"name":..}}`. Normalised to a string (the function name for a named choice) by [`tool_choice_str`].
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// OpenAI's rename of `max_tokens`. Preferred when both are present.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Not in the OpenAI schema; llama.cpp/vLLM/Ollama all accept it and so do we.
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// `"\n"` or `["\n", "END"]` (OpenAI: up to 4).
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// llama.cpp extension (1.0 = off).
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
}

/// The validated, per-request generation config handed to [`ChatGenerator::chat`]. `None` fields
/// mean "the request didn't say" — the generator leaves the process default (`INFR_TEMP` /
/// `INFR_TOP_K` / `INFR_TOP_P` / `INFR_MAX_NEW`) in charge for exactly those.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repeat_penalty: Option<f32>,
    /// Already normalised: empty strings dropped (an empty stop would fire on the first token).
    pub stop: Vec<String>,
}

/// An OpenAI-shaped 400: `{"error":{"message":..,"type":"invalid_request_error","param":..}}`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamError {
    pub param: &'static str,
    pub message: String,
}

impl GenParams {
    /// Validate + normalise the request's sampling fields. Out-of-range values are a 400, NEVER a
    /// silent clamp and never a panic (OpenAI's own ranges: temperature 0..2, top_p 0..1,
    /// presence/frequency -2..2, at most 4 stop sequences).
    pub fn from_request(req: &ChatRequest) -> Result<Self, ParamError> {
        let rng = |param: &'static str,
                   v: Option<f32>,
                   lo: f32,
                   hi: f32|
         -> Result<Option<f32>, ParamError> {
            match v {
                Some(x) if !x.is_finite() || x < lo || x > hi => Err(ParamError {
                    param,
                    message: format!("{param} must be between {lo} and {hi}, got {x}"),
                }),
                other => Ok(other),
            }
        };

        let top_k = match req.top_k {
            // 0 = "no top-k" in llama.cpp; negative is meaningless.
            Some(k) if k < 0 => {
                return Err(ParamError {
                    param: "top_k",
                    message: format!("top_k must be >= 0, got {k}"),
                })
            }
            Some(k) => Some(k as usize),
            None => None,
        };

        let stop = match &req.stop {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(a)) => {
                let mut v = Vec::with_capacity(a.len());
                for item in a {
                    match item.as_str() {
                        Some(s) => v.push(s.to_string()),
                        None => {
                            return Err(ParamError {
                                param: "stop",
                                message: "stop must be a string or an array of strings".into(),
                            })
                        }
                    }
                }
                v
            }
            Some(_) => {
                return Err(ParamError {
                    param: "stop",
                    message: "stop must be a string or an array of strings".into(),
                })
            }
        };
        if stop.len() > 4 {
            return Err(ParamError {
                param: "stop",
                message: format!("at most 4 stop sequences are supported, got {}", stop.len()),
            });
        }
        // An empty stop string would match at every position — drop it rather than dead-stop.
        let stop: Vec<String> = stop.into_iter().filter(|s| !s.is_empty()).collect();

        if let Some(p) = req.repeat_penalty {
            if !p.is_finite() || p <= 0.0 {
                return Err(ParamError {
                    param: "repeat_penalty",
                    message: format!("repeat_penalty must be > 0, got {p}"),
                });
            }
        }

        Ok(Self {
            // OpenAI renamed `max_tokens` -> `max_completion_tokens`; the new name wins.
            max_tokens: req.max_completion_tokens.or(req.max_tokens),
            temperature: rng("temperature", req.temperature, 0.0, 2.0)?,
            top_p: rng("top_p", req.top_p, 0.0, 1.0)?,
            top_k,
            seed: req.seed,
            presence_penalty: rng("presence_penalty", req.presence_penalty, -2.0, 2.0)?,
            frequency_penalty: rng("frequency_penalty", req.frequency_penalty, -2.0, 2.0)?,
            repeat_penalty: req.repeat_penalty,
            stop,
        })
    }
}

// ---------------------------------------------------------------------------
// Stop sequences
// ---------------------------------------------------------------------------

/// Incremental stop-sequence matcher over the DECODED text stream.
///
/// The hard part is that a stop string need not align with token boundaries: `"\n\n"` may arrive as
/// `"\n"` + `"\n"`, and `"END"` as `"E"` + `"ND"`. Two rules make that work:
///
/// 1. **Match on the accumulated tail**, not on the individual piece — so a split stop still fires.
/// 2. **Hold back** the longest suffix of the emitted text that is a PREFIX of some stop string.
///    Streaming clients must never see `"E"` from a token that turns out to begin `"END"`; if the
///    next piece completes the stop, that `"E"` was never ours to send. The hold-back is bounded by
///    `longest_stop - 1` bytes, so at most 3-4 bytes of latency in practice.
///
/// On a hit, the text BEFORE the stop string is emitted and the stop string itself is discarded
/// (OpenAI does not include it in the completion).
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    /// Text seen but not yet emitted (a possible stop prefix).
    held: String,
    hit: bool,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            held: String::new(),
            hit: false,
        }
    }

    pub fn is_active(&self) -> bool {
        !self.stops.is_empty()
    }

    /// A stop sequence has fired: generation must halt and no further text may be emitted.
    pub fn hit(&self) -> bool {
        self.hit
    }

    /// Feed one decoded piece; returns the text that is now SAFE to emit (possibly empty).
    pub fn push(&mut self, piece: &str) -> String {
        if self.hit {
            return String::new();
        }
        if self.stops.is_empty() {
            return piece.to_string();
        }
        self.held.push_str(piece);

        // 1. Full match anywhere in the held tail -> emit the head, drop the stop and the rest.
        if let Some(cut) = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()))
            .min()
        {
            self.hit = true;
            let out = self.held[..cut].to_string();
            self.held.clear();
            return out;
        }

        // 2. No match: hold back the longest suffix that could still BECOME one.
        let hold = self.longest_partial_suffix();
        let split = self.held.len() - hold;
        let out = self.held[..split].to_string();
        self.held.drain(..split);
        out
    }

    /// End of generation with no stop hit: whatever is still held was never a stop, so emit it.
    pub fn flush(&mut self) -> String {
        if self.hit {
            return String::new();
        }
        std::mem::take(&mut self.held)
    }

    /// Length (bytes) of the longest suffix of `held` that is a proper prefix of some stop string.
    /// Always lands on a char boundary: a suffix of `held` equal to `stop[..n]` starts at `stop`'s
    /// first byte, which is a UTF-8 lead byte.
    fn longest_partial_suffix(&self) -> usize {
        let max = self
            .stops
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
            .saturating_sub(1)
            .min(self.held.len());
        for n in (1..=max).rev() {
            let start = self.held.len() - n;
            if !self.held.is_char_boundary(start) {
                continue;
            }
            let tail = &self.held[start..];
            if self
                .stops
                .iter()
                .any(|s| s.as_bytes().starts_with(tail.as_bytes()))
            {
                return n;
            }
        }
        0
    }
}

/// Normalise OpenAI `tool_choice` to a string the generator understands: `"auto"`/`"required"`/
/// `"none"` pass through; a `{"type":"function","function":{"name":N}}` object becomes `N`.
fn tool_choice_str(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(_) => v
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .map(str::to_owned),
        _ => None,
    }
}

/// A single chat message.  `content` may be a JSON string or a content-part array.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessageDto {
    pub role: String,
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    /// Assistant's prior tool calls (OpenAI `[{id,type,function:{name,arguments}}]`), replayed on the
    /// next turn so the model sees its own calls.
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

// ---------------------------------------------------------------------------
// Response DTOs — /v1/models
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

// ---------------------------------------------------------------------------
// Response DTOs — /v1/chat/completions  (non-streaming)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OAIToolCall>>,
}

/// OpenAI-shaped tool call (used in both streaming and non-streaming responses).
#[derive(Debug, Clone, Serialize)]
pub struct OAIToolCall {
    pub index: usize,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: OAIFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct OAIFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// Response DTOs — /v1/chat/completions  (streaming chunks)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: DeltaPayload,
    pub finish_reason: Option<String>,
}

/// The `delta` field inside a streaming chunk.  Fields absent when `None`.
#[derive(Debug, Default, Serialize)]
pub struct DeltaPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OAIToolCall>>,
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

/// Shared server state.
///
/// `engine` is an `Option<Arc<dyn ChatGenerator>>` so the router can be constructed without a live
/// model (used in tests and for the /health + /v1/models endpoints which need no generation). It is
/// an `Arc`, NOT an `Arc<Mutex<_>>`: generation runs concurrently and the generator synchronises
/// itself (see [`ChatGenerator`]'s `&self` contract).
///
/// `slots` is the ADMISSION control — one permit per KV slot (`--parallel N`). A request takes a
/// permit for the whole of its generation and returns it at the end, so at most N generate
/// concurrently and the N+1'th QUEUES (tokio's semaphore is FIFO, so it queues fairly) rather than
/// being rejected. This is the only thing that bounds in-flight work; the generator's own slot pool
/// is sized to match.
#[derive(Clone)]
pub struct AppState {
    pub engine: Option<Arc<dyn ChatGenerator>>,
    pub model_id: Arc<str>,
    slots: Arc<Semaphore>,
}

impl AppState {
    /// Wrap a loaded generator for production use with `n_parallel` concurrent slots.
    pub fn new(
        generator: Arc<dyn ChatGenerator>,
        model_id: impl Into<String>,
        n_parallel: usize,
    ) -> Self {
        Self {
            engine: Some(generator),
            model_id: Arc::from(model_id.into().as_str()),
            slots: Arc::new(Semaphore::new(n_parallel.max(1))),
        }
    }

    /// No-engine state — for /health, /v1/models, and serialisation tests.
    pub fn headless(model_id: impl Into<String>) -> Self {
        Self {
            engine: None,
            model_id: Arc::from(model_id.into().as_str()),
            slots: Arc::new(Semaphore::new(1)),
        }
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the axum [`Router`].  Extracted so tests can call it with a [`AppState::headless`] state.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

/// Start the OpenAI-compatible server bound to `addr`, serving `engine` under `model_id` with
/// `n_parallel` concurrent generation slots (`--parallel N`).
pub async fn serve(
    generator: Arc<dyn ChatGenerator>,
    model_id: String,
    addr: SocketAddr,
    n_parallel: usize,
) -> anyhow::Result<()> {
    let state = AppState::new(generator, model_id, n_parallel);
    let router = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, %n_parallel, "infr-server listening");
    axum::serve(listener, router).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health_handler() -> StatusCode {
    StatusCode::OK
}

async fn models_handler(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelCard {
            id: state.model_id.to_string(),
            object: "model",
            owned_by: "local",
        }],
    })
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    body: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    // Malformed JSON / wrong types: an OpenAI-shaped 400, not axum's default 422 text body.
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return param_error(None, e.body_text()),
    };
    let params = match GenParams::from_request(&req) {
        Ok(p) => p,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    let messages: Vec<ChatMessage> = req.messages.iter().map(dto_to_engine).collect();
    let tools_json: Option<String> = req.tools.as_ref().map(|v| v.to_string());
    let tool_choice: Option<String> = req.tool_choice.as_ref().and_then(tool_choice_str);
    let model_id = state.model_id.to_string();
    let cid = make_id();
    let created = unix_ts();

    if req.stream {
        streaming(
            state,
            messages,
            tools_json,
            tool_choice,
            params,
            cid,
            model_id,
            created,
        )
        .await
    } else {
        non_streaming(
            state,
            messages,
            tools_json,
            tool_choice,
            params,
            cid,
            model_id,
            created,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Non-streaming path
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn non_streaming(
    state: AppState,
    messages: Vec<ChatMessage>,
    tools_json: Option<String>,
    tool_choice: Option<String>,
    params: GenParams,
    cid: String,
    model_id: String,
    created: i64,
) -> Response {
    // Wait for a free slot. With `--parallel N`, the (N+1)'th concurrent request queues HERE — in
    // the async runtime, holding no thread — and is admitted FIFO as soon as one finishes.
    let Ok(permit) = state.slots.clone().acquire_owned().await else {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".into(),
        );
    };
    let engine_arc = state.engine.clone();
    let cid_blk = cid.clone();

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        // The permit is MOVED into the blocking task and dropped when it ends: a slot is held for
        // exactly the generation, and the next queued request is admitted the moment it frees.
        let _permit = permit;
        let Some(engine) = engine_arc else {
            anyhow::bail!("no engine loaded");
        };

        let mut reasoning = String::new();
        let mut content = String::new();
        let mut tool_calls: Vec<OAIToolCall> = Vec::new();

        let finish = engine
            .chat(
                &messages,
                tools_json.as_deref(),
                tool_choice.as_deref(),
                &params,
                &mut |delta| match delta {
                    Delta::Reasoning(t) => reasoning.push_str(&t),
                    Delta::Content(t) => content.push_str(&t),
                    Delta::ToolCall { name, arguments } => {
                        let idx = tool_calls.len();
                        tool_calls.push(OAIToolCall {
                            index: idx,
                            id: format!("call_{cid_blk}_{idx}"),
                            kind: "function",
                            function: OAIFunction { name, arguments },
                        });
                    }
                },
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok((reasoning, content, tool_calls, finish))
    })
    .await
    .map_err(anyhow::Error::from)
    .and_then(|r| r);

    match result {
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Ok((reasoning, content, tool_calls, finish)) => {
            let finish = if tool_calls.is_empty() {
                finish
            } else {
                Finish::ToolCalls
            }
            .as_str();
            Json(ChatCompletionResponse {
                id: cid,
                object: "chat.completion",
                created,
                model: model_id,
                choices: vec![CompletionChoice {
                    index: 0,
                    message: AssistantMessage {
                        role: "assistant",
                        content: if content.is_empty() {
                            None
                        } else {
                            Some(content.clone())
                        },
                        reasoning_content: if reasoning.is_empty() {
                            None
                        } else {
                            Some(reasoning)
                        },
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(tool_calls)
                        },
                    },
                    finish_reason: finish.into(),
                }],
                usage: UsageInfo {
                    prompt_tokens: 0,
                    completion_tokens: (content.len() / 4) as u32,
                    total_tokens: 0,
                },
            })
            .into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming path (SSE)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn streaming(
    state: AppState,
    messages: Vec<ChatMessage>,
    tools_json: Option<String>,
    tool_choice: Option<String>,
    params: GenParams,
    cid: String,
    model_id: String,
    created: i64,
) -> Response {
    // UNBOUNDED on purpose. The generator's `on_delta` callback is invoked from inside the decode
    // loop — which, under `--parallel N`, is holding the GPU baton. A bounded channel would make a
    // slow (or stalled) SSE client apply backpressure straight into that callback, so ONE
    // non-draining client would block the GPU step it is inside of and stall every OTHER sequence
    // behind it: precisely the head-of-line blocking this whole change exists to remove. Decoupling
    // the socket from the decode loop costs a queue whose depth is bounded anyway by `max_tokens`
    // (a few thousand short strings, worst case), which is the right trade.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();

    // Same admission gate as the non-streaming path — the (N+1)'th concurrent stream queues here.
    // Taken BEFORE the SSE response is returned, so a queued client simply waits for its first byte
    // rather than being handed an open-but-silent stream.
    let Ok(permit) = state.slots.clone().acquire_owned().await else {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".into(),
        );
    };
    let engine_arc = state.engine.clone();
    // Clone sender + strings for use inside the on_delta callback closure.
    let tx_cb = tx.clone();
    let cid_cb = cid.clone();
    let model_cb = model_id.clone();

    tokio::task::spawn_blocking(move || {
        // Held for exactly this generation; freed for the next queued request on return.
        let _permit = permit;

        // First chunk: role delta (mirrors the Python shim's opening chunk).
        let _ = tx.send(Ok(sse_chunk(
            &cid,
            &model_id,
            created,
            DeltaPayload {
                role: Some("assistant".into()),
                ..Default::default()
            },
            None,
        )));

        let Some(engine) = engine_arc else {
            // No engine — close the stream immediately.
            let _ = tx.send(Ok(Event::default().data("[DONE]")));
            return;
        };

        let mut tc_index = 0usize;
        let mut saw_tool_call = false;

        let res = engine.chat(
            &messages,
            tools_json.as_deref(),
            tool_choice.as_deref(),
            &params,
            &mut |delta| {
                let payload = match delta {
                    Delta::Reasoning(t) => DeltaPayload {
                        reasoning_content: Some(t),
                        ..Default::default()
                    },
                    Delta::Content(t) => DeltaPayload {
                        content: Some(t),
                        ..Default::default()
                    },
                    Delta::ToolCall { name, arguments } => {
                        let tc = OAIToolCall {
                            index: tc_index,
                            id: format!("call_{cid_cb}_{tc_index}"),
                            kind: "function",
                            function: OAIFunction { name, arguments },
                        };
                        tc_index += 1;
                        saw_tool_call = true;
                        DeltaPayload {
                            tool_calls: Some(vec![tc]),
                            ..Default::default()
                        }
                    }
                };
                let _ = tx_cb.send(Ok(sse_chunk(&cid_cb, &model_cb, created, payload, None)));
            },
        );

        let finish = if saw_tool_call {
            Finish::ToolCalls
        } else {
            res.unwrap_or(Finish::Stop)
        };

        // Finish chunk: empty delta + finish_reason.
        let _ = tx.send(Ok(sse_chunk(
            &cid,
            &model_id,
            created,
            DeltaPayload::default(),
            Some(finish.as_str().into()),
        )));

        // OpenAI SSE sentinel.
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
        // `tx` and `tx_cb` are both dropped here, which closes the channel and ends the stream.
    });

    // Bridge the mpsc receiver to an async Stream for axum's Sse.
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });

    Sse::new(stream).into_response()
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Serialize a delta payload into an SSE event carrying a `chat.completion.chunk`.
fn sse_chunk(
    cid: &str,
    model: &str,
    created: i64,
    delta: DeltaPayload,
    finish_reason: Option<String>,
) -> Event {
    let chunk = ChatCompletionChunk {
        id: cid.to_owned(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
    };
    Event::default()
        .json_data(chunk)
        .expect("ChatCompletionChunk always serializes")
}

fn json_error(status: StatusCode, msg: String) -> Response {
    let body = serde_json::json!({"error": {"message": msg, "type": "server_error"}});
    (status, Json(body)).into_response()
}

/// OpenAI-shaped 400 for a bad request parameter (`invalid_request_error`, with the offending
/// `param` named). NOT a clamp and NOT a panic — see [`GenParams::from_request`].
fn param_error(param: Option<&str>, msg: String) -> Response {
    let body = serde_json::json!({"error": {
        "message": msg,
        "type": "invalid_request_error",
        "param": param,
        "code": serde_json::Value::Null,
    }});
    (StatusCode::BAD_REQUEST, Json(body)).into_response()
}

fn make_id() -> String {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("chatcmpl-{ms}")
}

fn unix_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Flatten a DTO `content` field (string OR content-part array) to a plain `String`.
///
/// Mirrors the Python shim's `normalize_messages`: only `"text"` parts are kept.
pub fn flatten_content(v: &Option<serde_json::Value>) -> String {
    match v {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                if p.get("type")?.as_str()? == "text" {
                    p.get("text")?.as_str().map(str::to_owned)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(other) => other.to_string(),
    }
}

fn dto_to_engine(dto: &ChatMessageDto) -> ChatMessage {
    ChatMessage {
        role: dto.role.clone(),
        content: flatten_content(&dto.content),
        tool_calls: dto.tool_calls.as_ref().and_then(parse_oai_tool_calls),
        tool_call_id: dto.tool_call_id.clone(),
        name: dto.name.clone(),
    }
}

/// Convert an inbound OpenAI `tool_calls` array (`[{function:{name, arguments}}]`, where `arguments`
/// is a JSON STRING) into engine [`ToolCall`]s with `arguments` parsed to a `Value`. Returns `None`
/// if the field isn't a non-empty array of valid calls.
fn parse_oai_tool_calls(v: &serde_json::Value) -> Option<Vec<ToolCall>> {
    let arr = v.as_array()?;
    let calls: Vec<ToolCall> = arr
        .iter()
        .filter_map(|c| {
            let f = c.get("function")?;
            let name = f.get("name")?.as_str()?.to_owned();
            let arguments = match f.get("arguments") {
                Some(serde_json::Value::String(s)) => {
                    serde_json::from_str(s).unwrap_or(serde_json::Value::String(s.clone()))
                }
                Some(other) => other.clone(),
                None => serde_json::Value::Object(serde_json::Map::new()),
            };
            Some(ToolCall { name, arguments })
        })
        .collect();
    (!calls.is_empty()).then_some(calls)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// Router backed by a headless state — no Engine, so /health and /v1/models work
    /// but /v1/chat/completions would return 500.  That's fine: we never call it here.
    fn test_router() -> Router {
        build_router(AppState::headless("test-model"))
    }

    // --- HTTP endpoint tests (no Engine required) ---------------------------

    #[tokio::test]
    async fn test_health_returns_200() {
        let resp = test_router()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_models_returns_200_with_expected_shape() {
        let resp = test_router()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["object"], "list");
        let card = &json["data"][0];
        assert_eq!(card["id"], "test-model");
        assert_eq!(card["object"], "model");
        assert_eq!(card["owned_by"], "local");
    }

    // --- ChatRequest serde round-trip tests --------------------------------

    #[test]
    fn chat_request_string_content_deserializes() {
        let raw = r#"{"model":"m","messages":[{"role":"user","content":"hello"}]}"#;
        let req: ChatRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(flatten_content(&req.messages[0].content), "hello");
        assert!(!req.stream);
        assert!(req.tools.is_none());
        assert!(req.max_tokens.is_none());
    }

    #[test]
    fn chat_request_array_content_deserializes_and_flattens() {
        let raw = r#"{
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type":"text","text":"hello"},
                    {"type":"image_url","image_url":{"url":"data:..."}},
                    {"type":"text","text":" world"}
                ]
            }]
        }"#;
        let req: ChatRequest = serde_json::from_str(raw).unwrap();
        // Only text parts are concatenated; image_url is discarded.
        assert_eq!(flatten_content(&req.messages[0].content), "hello world");
    }

    #[test]
    fn chat_request_stream_flag() {
        let raw = r#"{"model":"m","messages":[],"stream":true}"#;
        let req: ChatRequest = serde_json::from_str(raw).unwrap();
        assert!(req.stream);
    }

    #[test]
    fn chat_request_with_tools_and_max_tokens() {
        let raw = r#"{
            "model":"m","messages":[],
            "tools":[{"type":"function","function":{"name":"bash","description":"run bash"}}],
            "max_tokens":512
        }"#;
        let req: ChatRequest = serde_json::from_str(raw).unwrap();
        assert!(req.tools.is_some());
        assert_eq!(req.max_tokens, Some(512));
    }

    #[test]
    fn chat_message_dto_with_tool_call_id_and_name() {
        let raw = r#"{"role":"tool","content":"result","tool_call_id":"tc_1","name":"bash"}"#;
        let msg: ChatMessageDto = serde_json::from_str(raw).unwrap();
        assert_eq!(msg.tool_call_id.as_deref(), Some("tc_1"));
        assert_eq!(msg.name.as_deref(), Some("bash"));
    }

    // --- Non-streaming response serialization tests ------------------------

    #[test]
    fn chat_completion_response_stop_serializes() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".into(),
            object: "chat.completion",
            created: 1000,
            model: "test-model".into(),
            choices: vec![CompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: Some("hello".into()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: UsageInfo {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "hello");
        // skip_serializing_if = None → field absent in JSON → serde_json gives Null on access
        assert!(v["choices"][0]["message"]["reasoning_content"].is_null());
        assert!(v["choices"][0]["message"]["tool_calls"].is_null());
    }

    #[test]
    fn chat_completion_response_with_reasoning_content() {
        let resp = ChatCompletionResponse {
            id: "id".into(),
            object: "chat.completion",
            created: 0,
            model: "m".into(),
            choices: vec![CompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: Some("answer".into()),
                    reasoning_content: Some("I thought about it".into()),
                    tool_calls: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: UsageInfo {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(
            v["choices"][0]["message"]["reasoning_content"],
            "I thought about it"
        );
        assert_eq!(v["choices"][0]["message"]["content"], "answer");
    }

    #[test]
    fn chat_completion_response_tool_calls_finish_reason() {
        let resp = ChatCompletionResponse {
            id: "id".into(),
            object: "chat.completion",
            created: 0,
            model: "m".into(),
            choices: vec![CompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![OAIToolCall {
                        index: 0,
                        id: "call_0".into(),
                        kind: "function",
                        function: OAIFunction {
                            name: "bash".into(),
                            arguments: r#"{"command":"ls"}"#.into(),
                        },
                    }]),
                },
                finish_reason: "tool_calls".into(),
            }],
            usage: UsageInfo {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        let tc = &v["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "bash");
        assert_eq!(tc["function"]["arguments"], r#"{"command":"ls"}"#);
        assert_eq!(tc["index"], 0);
        // content: None serializes as null
        assert!(v["choices"][0]["message"]["content"].is_null());
    }

    // --- Streaming chunk serialization tests --------------------------------

    #[test]
    fn streaming_chunk_role_delta() {
        let chunk = ChatCompletionChunk {
            id: "id".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaPayload {
                    role: Some("assistant".into()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        // Other delta fields absent
        assert!(v["choices"][0]["delta"]["content"].is_null());
        assert!(v["choices"][0]["delta"]["reasoning_content"].is_null());
        assert!(v["choices"][0]["delta"]["tool_calls"].is_null());
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn streaming_chunk_content_delta() {
        let chunk = ChatCompletionChunk {
            id: "id".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaPayload {
                    content: Some("hello".into()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], "hello");
        assert!(v["choices"][0]["delta"]["role"].is_null());
    }

    #[test]
    fn streaming_chunk_reasoning_content_delta() {
        let chunk = ChatCompletionChunk {
            id: "id".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaPayload {
                    reasoning_content: Some("thinking...".into()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(v["choices"][0]["delta"]["reasoning_content"], "thinking...");
        assert!(v["choices"][0]["delta"]["content"].is_null());
    }

    #[test]
    fn streaming_chunk_tool_call_delta() {
        let chunk = ChatCompletionChunk {
            id: "id".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaPayload {
                    tool_calls: Some(vec![OAIToolCall {
                        index: 0,
                        id: "call_0".into(),
                        kind: "function",
                        function: OAIFunction {
                            name: "bash".into(),
                            arguments: r#"{"cmd":"ls"}"#.into(),
                        },
                    }]),
                    ..Default::default()
                },
                finish_reason: None,
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        let tc = &v["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["index"], 0);
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "bash");
    }

    #[test]
    fn streaming_chunk_finish_reason() {
        let chunk = ChatCompletionChunk {
            id: "id".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaPayload::default(), // empty delta {}
                finish_reason: Some("stop".into()),
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        // Empty delta: all fields None → absent in JSON
        assert!(v["choices"][0]["delta"]["content"].is_null());
        assert!(v["choices"][0]["delta"]["role"].is_null());
    }

    // --- /v1/models serialization test ------------------------------------

    #[test]
    fn models_response_serializes() {
        let resp = ModelsResponse {
            object: "list",
            data: vec![ModelCard {
                id: "my-model".into(),
                object: "model",
                owned_by: "local",
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "my-model");
        assert_eq!(v["data"][0]["object"], "model");
        assert_eq!(v["data"][0]["owned_by"], "local");
    }

    // --- sampling param plumbing -------------------------------------------

    fn req(raw: &str) -> ChatRequest {
        serde_json::from_str(raw).unwrap()
    }

    #[test]
    fn absent_sampling_fields_stay_none() {
        // The whole point of Option-everything: a request that says nothing must not override the
        // process defaults (INFR_TEMP/TOP_K/TOP_P), i.e. today's behavior is preserved exactly.
        let p = GenParams::from_request(&req(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
        assert_eq!(p, GenParams::default());
    }

    #[test]
    fn sampling_fields_parse() {
        let p = GenParams::from_request(&req(
            r#"{"model":"m","messages":[],"temperature":0.0,"top_p":0.9,"top_k":40,"seed":42,
                "presence_penalty":0.5,"frequency_penalty":-0.25,"repeat_penalty":1.1,
                "stop":["\n\n","END"]}"#,
        ))
        .unwrap();
        assert_eq!(p.temperature, Some(0.0));
        assert_eq!(p.top_p, Some(0.9));
        assert_eq!(p.top_k, Some(40));
        assert_eq!(p.seed, Some(42));
        assert_eq!(p.presence_penalty, Some(0.5));
        assert_eq!(p.frequency_penalty, Some(-0.25));
        assert_eq!(p.repeat_penalty, Some(1.1));
        assert_eq!(p.stop, vec!["\n\n".to_string(), "END".to_string()]);
    }

    #[test]
    fn unknown_fields_are_ignored_not_rejected() {
        let p = GenParams::from_request(&req(
            r#"{"model":"m","messages":[],"n":1,"user":"bob","logit_bias":{},"logprobs":true}"#,
        ))
        .unwrap();
        assert_eq!(p, GenParams::default());
    }

    #[test]
    fn max_completion_tokens_is_preferred_alias() {
        let p = GenParams::from_request(&req(r#"{"model":"m","messages":[],"max_tokens":10}"#))
            .unwrap();
        assert_eq!(p.max_tokens, Some(10));
        let p = GenParams::from_request(&req(
            r#"{"model":"m","messages":[],"max_completion_tokens":20}"#,
        ))
        .unwrap();
        assert_eq!(p.max_tokens, Some(20));
        // Both present: the new name wins.
        let p = GenParams::from_request(&req(
            r#"{"model":"m","messages":[],"max_tokens":10,"max_completion_tokens":20}"#,
        ))
        .unwrap();
        assert_eq!(p.max_tokens, Some(20));
    }

    #[test]
    fn stop_accepts_a_bare_string() {
        let p =
            GenParams::from_request(&req(r#"{"model":"m","messages":[],"stop":"\n"}"#)).unwrap();
        assert_eq!(p.stop, vec!["\n".to_string()]);
    }

    #[test]
    fn empty_stop_strings_are_dropped() {
        // An empty stop would match at position 0 of every step — kill it at the door.
        let p =
            GenParams::from_request(&req(r#"{"model":"m","messages":[],"stop":["",""]}"#)).unwrap();
        assert!(p.stop.is_empty());
    }

    #[test]
    fn invalid_values_are_param_errors_not_clamps() {
        for (raw, param) in [
            (
                r#"{"model":"m","messages":[],"temperature":-1}"#,
                "temperature",
            ),
            (
                r#"{"model":"m","messages":[],"temperature":3}"#,
                "temperature",
            ),
            (r#"{"model":"m","messages":[],"top_p":5}"#, "top_p"),
            (r#"{"model":"m","messages":[],"top_k":-2}"#, "top_k"),
            (
                r#"{"model":"m","messages":[],"presence_penalty":-3}"#,
                "presence_penalty",
            ),
            (
                r#"{"model":"m","messages":[],"frequency_penalty":9}"#,
                "frequency_penalty",
            ),
            (
                r#"{"model":"m","messages":[],"repeat_penalty":0}"#,
                "repeat_penalty",
            ),
            (r#"{"model":"m","messages":[],"stop":[1,2]}"#, "stop"),
            (
                r#"{"model":"m","messages":[],"stop":["a","b","c","d","e"]}"#,
                "stop",
            ),
        ] {
            let e = GenParams::from_request(&req(raw)).unwrap_err();
            assert_eq!(e.param, param, "{raw}");
        }
    }

    #[tokio::test]
    async fn bad_temperature_returns_openai_shaped_400() {
        let resp = test_router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":-1}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "temperature");
    }

    // --- StopMatcher: the token-boundary case -------------------------------

    /// Drive a matcher over a token sequence, returning (emitted text, hit).
    fn run_stops(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()).collect());
        let mut out = String::new();
        for p in pieces {
            out.push_str(&m.push(p));
            if m.hit() {
                break;
            }
        }
        if !m.hit() {
            out.push_str(&m.flush());
        }
        (out, m.hit())
    }

    #[test]
    fn stop_within_one_token_fires_and_is_excluded() {
        let (out, hit) = run_stops(&["END"], &["hello ", "END", " more"]);
        assert!(hit);
        assert_eq!(out, "hello ");
    }

    #[test]
    fn stop_split_across_two_tokens_still_fires() {
        // THE boundary case: "END" arrives as "E" + "ND". It must fire, and the partial "E" must
        // NEVER have been emitted.
        let (out, hit) = run_stops(&["END"], &["hello ", "E", "ND", " more"]);
        assert!(hit, "a stop split across tokens must fire");
        assert_eq!(out, "hello ", "no partial stop prefix may leak");
    }

    #[test]
    fn stop_split_across_three_tokens_still_fires() {
        let (out, hit) = run_stops(&["<|done|>"], &["a", "<|", "do", "ne", "|>", "b"]);
        assert!(hit);
        assert_eq!(out, "a");
    }

    #[test]
    fn double_newline_stop_split_across_tokens() {
        let (out, hit) = run_stops(&["\n\n"], &["line", "\n", "\n", "next"]);
        assert!(hit);
        assert_eq!(out, "line");
    }

    #[test]
    fn stop_prefix_that_does_not_complete_is_eventually_emitted() {
        // "E" looked like the start of "END" but turned out to be "Every" — it must still be
        // delivered, exactly once, in order.
        let (out, hit) = run_stops(&["END"], &["E", "very", " day"]);
        assert!(!hit);
        assert_eq!(out, "Every day");
    }

    #[test]
    fn held_prefix_is_flushed_at_end_of_generation() {
        // Generation ends while the matcher still holds a partial prefix -> flush must emit it.
        let (out, hit) = run_stops(&["END"], &["done E"]);
        assert!(!hit);
        assert_eq!(out, "done E");
    }

    #[test]
    fn multiple_stops_take_the_earliest_match() {
        let (out, hit) = run_stops(&["World", "lo"], &["hel", "lo World"]);
        assert!(hit);
        assert_eq!(out, "hel");
    }

    #[test]
    fn no_stops_is_a_passthrough() {
        let (out, hit) = run_stops(&[], &["a", "b", "c"]);
        assert!(!hit);
        assert_eq!(out, "abc");
        assert!(!StopMatcher::new(vec![]).is_active());
    }

    #[test]
    fn multibyte_stop_split_mid_codepoint_is_safe() {
        // Pieces are always whole UTF-8, but a stop's own bytes may straddle them: "→END" arriving
        // as "→" + "END". Must fire without panicking on a char boundary.
        let (out, hit) = run_stops(&["→END"], &["x", "→", "END", "y"]);
        assert!(hit);
        assert_eq!(out, "x");
    }

    #[test]
    fn nothing_is_emitted_after_a_hit() {
        let mut m = StopMatcher::new(vec!["END".into()]);
        assert_eq!(m.push("aEND"), "a");
        assert!(m.hit());
        assert_eq!(m.push("more"), "");
        assert_eq!(m.flush(), "");
    }

    // --- finish_reason ------------------------------------------------------

    #[test]
    fn finish_reason_strings() {
        assert_eq!(Finish::Stop.as_str(), "stop");
        assert_eq!(Finish::Length.as_str(), "length");
        assert_eq!(Finish::ToolCalls.as_str(), "tool_calls");
    }

    // --- flatten_content unit tests ----------------------------------------

    #[test]
    fn flatten_content_string_value() {
        let v = Some(serde_json::Value::String("hello world".into()));
        assert_eq!(flatten_content(&v), "hello world");
    }

    #[test]
    fn flatten_content_array_skips_non_text_parts() {
        let v = Some(serde_json::json!([
            {"type": "text",      "text": "hello"},
            {"type": "image_url", "image_url": {"url": "http://x"}},
            {"type": "text",      "text": " world"}
        ]));
        assert_eq!(flatten_content(&v), "hello world");
    }

    #[test]
    fn flatten_content_none_gives_empty_string() {
        assert_eq!(flatten_content(&None), "");
    }

    #[test]
    fn flatten_content_null_json_gives_empty_string() {
        // An assistant tool-call message legally has `content: null` — it must flatten to "" (not the
        // literal "null", which would inject a stray word into the prompt).
        assert_eq!(flatten_content(&Some(serde_json::Value::Null)), "");
    }
}
