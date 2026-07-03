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
use infr_engine::{ChatMessage, Delta, Engine, ToolCall};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

/// Generation backend the server drives — it never knows the model/GPU underneath.
/// Implemented by `Engine` (below) and by CLI adapters (e.g. a Llama runner).
/// `max_tokens` is the request's generation budget (OpenAI `max_tokens`); `None` leaves the
/// generator's own default in charge.
pub trait ChatGenerator: Send {
    fn chat(
        &mut self,
        messages: &[ChatMessage],
        tools_json: Option<&str>,
        tool_choice: Option<&str>,
        max_tokens: Option<u32>,
        on_delta: &mut dyn FnMut(Delta),
    ) -> anyhow::Result<()>;
}

impl ChatGenerator for Engine {
    fn chat(
        &mut self,
        messages: &[ChatMessage],
        tools_json: Option<&str>,
        _tool_choice: Option<&str>,
        _max_tokens: Option<u32>,
        on_delta: &mut dyn FnMut(Delta),
    ) -> anyhow::Result<()> {
        Engine::chat(self, messages, tools_json, on_delta).map_err(|e| anyhow::anyhow!("{e}"))
    }
}

// ---------------------------------------------------------------------------
// Request DTOs
// ---------------------------------------------------------------------------

/// Top-level chat completion request (OpenAI wire format).
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
/// `engine` is `Option<Engine>` so the router can be constructed without a live model
/// (used in tests and for the /health + /v1/models endpoints which need no generation).
/// Generation is serialised by the Mutex — single-stream for the MVP.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Option<Box<dyn ChatGenerator>>>>,
    pub model_id: Arc<str>,
}

impl AppState {
    /// Wrap a loaded generator for production use.
    pub fn new(generator: Box<dyn ChatGenerator>, model_id: impl Into<String>) -> Self {
        Self {
            engine: Arc::new(Mutex::new(Some(generator))),
            model_id: Arc::from(model_id.into().as_str()),
        }
    }

    /// No-engine state — for /health, /v1/models, and serialisation tests.
    pub fn headless(model_id: impl Into<String>) -> Self {
        Self {
            engine: Arc::new(Mutex::new(None)),
            model_id: Arc::from(model_id.into().as_str()),
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

/// Start the OpenAI-compatible server bound to `addr`, serving `engine` under `model_id`.
pub async fn serve(
    generator: Box<dyn ChatGenerator>,
    model_id: String,
    addr: SocketAddr,
) -> anyhow::Result<()> {
    let state = AppState::new(generator, model_id);
    let router = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "infr-server listening");
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
    Json(req): Json<ChatRequest>,
) -> Response {
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
            req.max_tokens,
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
            req.max_tokens,
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
    max_tokens: Option<u32>,
    cid: String,
    model_id: String,
    created: i64,
) -> Response {
    let engine_arc = state.engine.clone();
    let cid_blk = cid.clone();

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let handle = tokio::runtime::Handle::current();
        let mut guard = handle.block_on(engine_arc.lock());

        let Some(ref mut engine) = *guard else {
            anyhow::bail!("no engine loaded");
        };

        let mut reasoning = String::new();
        let mut content = String::new();
        let mut tool_calls: Vec<OAIToolCall> = Vec::new();

        engine
            .chat(
                &messages,
                tools_json.as_deref(),
                tool_choice.as_deref(),
                max_tokens,
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

        Ok((reasoning, content, tool_calls))
    })
    .await
    .map_err(anyhow::Error::from)
    .and_then(|r| r);

    match result {
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Ok((reasoning, content, tool_calls)) => {
            let finish = if tool_calls.is_empty() {
                "stop"
            } else {
                "tool_calls"
            };
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
    max_tokens: Option<u32>,
    cid: String,
    model_id: String,
    created: i64,
) -> Response {
    // Bounded channel: 64 events of headroom before the generator blocks.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    let engine_arc = state.engine.clone();
    // Clone sender + strings for use inside the on_delta callback closure.
    let tx_cb = tx.clone();
    let cid_cb = cid.clone();
    let model_cb = model_id.clone();

    tokio::task::spawn_blocking(move || {
        // Acquire the tokio runtime handle inherited from the parent async context.
        let handle = tokio::runtime::Handle::current();
        let mut guard = handle.block_on(engine_arc.lock());

        // First chunk: role delta (mirrors the Python shim's opening chunk).
        let _ = handle.block_on(tx.send(Ok(sse_chunk(
            &cid,
            &model_id,
            created,
            DeltaPayload {
                role: Some("assistant".into()),
                ..Default::default()
            },
            None,
        ))));

        let Some(ref mut engine) = *guard else {
            // No engine — close the stream immediately.
            let _ = handle.block_on(tx.send(Ok(Event::default().data("[DONE]"))));
            return;
        };

        let mut tc_index = 0usize;
        let mut finish: &str = "stop";

        let _ = engine.chat(
            &messages,
            tools_json.as_deref(),
            tool_choice.as_deref(),
            max_tokens,
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
                        finish = "tool_calls";
                        DeltaPayload {
                            tool_calls: Some(vec![tc]),
                            ..Default::default()
                        }
                    }
                };
                let _ = handle.block_on(
                    tx_cb.send(Ok(sse_chunk(&cid_cb, &model_cb, created, payload, None))),
                );
            },
        );

        // Finish chunk: empty delta + finish_reason.
        let _ = handle.block_on(tx.send(Ok(sse_chunk(
            &cid,
            &model_id,
            created,
            DeltaPayload::default(),
            Some(finish.into()),
        ))));

        // OpenAI SSE sentinel.
        let _ = handle.block_on(tx.send(Ok(Event::default().data("[DONE]"))));
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
