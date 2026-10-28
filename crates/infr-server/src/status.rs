//! `GET /status`: one JSON snapshot of the running server — each hosted model's backend, device,
//! context window, slots and device memory, plus the request gauges. It answers what an OpenAI
//! client cannot ask: where is this model running, and how much room is left there.
//!
//! Every value is read when the request arrives. Fields a backend cannot report are `null`, never
//! a zero that would read as a measurement.

use axum::{
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use std::sync::atomic::Ordering;

use crate::{auth_gate, AppState};

/// Where a generator runs and how much room it has there, as [`crate::ChatGenerator::status`]
/// reports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GeneratorStatus {
    /// The backend serving the model: `"vulkan"`, `"cpu"` or `"metal"`.
    pub backend: &'static str,
    /// The physical device's name, when the backend binds one.
    pub device: Option<String>,
    /// The context window each slot holds, in tokens.
    pub ctx_per_slot: Option<usize>,
    /// The device's memory as this process sees it.
    pub memory: Option<DeviceMemory>,
}

/// A device memory snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceMemory {
    pub total_bytes: u64,
    /// What this process could still allocate there, by the driver's own budget, which accounts
    /// for other processes' use.
    pub available_bytes: u64,
    /// `false` when the driver has no live budget to report, so `available_bytes` is only the
    /// heap size and does not move as memory is used.
    pub live: bool,
    /// The device shares system memory (an integrated GPU), so both figures cover that pool.
    pub unified: bool,
}

#[derive(Serialize)]
struct StatusResponse {
    version: &'static str,
    uptime_seconds: f64,
    requests: Requests,
    models: Vec<ModelStatus>,
}

#[derive(Serialize)]
struct Requests {
    running: u64,
    waiting: u64,
}

#[derive(Serialize)]
struct ModelStatus {
    id: String,
    slots: Slots,
    /// Absent only for a state built without a generator (tests).
    #[serde(flatten)]
    generator: Option<GeneratorStatus>,
}

#[derive(Serialize)]
struct Slots {
    total: usize,
    busy: usize,
}

/// Gated by the API key like `/v1/models`: it names every model and the hardware under them.
pub(crate) async fn status_handler(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    Json(StatusResponse {
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds: state.started.elapsed().as_secs_f64(),
        requests: Requests {
            running: state.stats.active.load(Ordering::Relaxed),
            waiting: state.stats.queued.load(Ordering::Relaxed),
        },
        models: state
            .models
            .iter()
            .map(|m| ModelStatus {
                id: m.id.to_string(),
                slots: Slots {
                    total: m.capacity,
                    busy: m.busy(),
                },
                generator: m.engine.as_ref().map(|e| e.status()),
            })
            .collect(),
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use crate::{build_router, AppState, ChatGenerator, Config};
    use axum::{body::Body, http::Request, http::StatusCode};
    use http_body_util::BodyExt;
    use std::sync::Arc;
    use tower::ServiceExt;

    async fn get_status(state: AppState, auth: Option<&str>) -> (StatusCode, serde_json::Value) {
        let mut b = Request::builder().uri("/status");
        if let Some(a) = auth {
            b = b.header("authorization", a);
        }
        let resp = build_router(state)
            .oneshot(b.body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    /// Each model carries its own generator's report and its live slot occupancy.
    #[tokio::test]
    async fn every_model_reports_its_generator_and_slots() {
        let a: Arc<dyn ChatGenerator> = Arc::new(crate::tests::EchoGen("alpha"));
        let b: Arc<dyn ChatGenerator> = Arc::new(crate::tests::EchoGen("beta"));
        let state = AppState::multi(
            vec![("alpha".into(), a, 2), ("beta".into(), b, 1)],
            Arc::new(Config::default()),
        );
        let _held = state.models[1].slots.clone().acquire_owned().await.unwrap();

        let (code, v) = get_status(state, None).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
        assert!(v["uptime_seconds"].as_f64().unwrap() >= 0.0);
        assert_eq!(v["requests"]["running"], 0);
        let models = v["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0]["id"], "alpha");
        assert_eq!(models[0]["device"], "dev-alpha");
        assert_eq!(models[0]["backend"], "vulkan");
        assert_eq!(models[0]["ctx_per_slot"], 4096);
        assert_eq!(models[0]["memory"]["available_bytes"], 6u64 << 30);
        assert_eq!(models[0]["slots"]["total"], 2);
        assert_eq!(models[0]["slots"]["busy"], 0);
        assert_eq!(models[1]["device"], "dev-beta");
        assert_eq!(models[1]["slots"]["busy"], 1);
    }

    #[tokio::test]
    async fn status_needs_the_api_key_when_one_is_set() {
        let mut cfg = Config::default();
        cfg.serve.api_key = Some("s3cret".into());
        let state = AppState::headless("hidden-model", Arc::new(cfg));

        let (code, v) = get_status(state.clone(), None).await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);
        assert!(!v.to_string().contains("hidden-model"));

        let (code, v) = get_status(state, Some("Bearer s3cret")).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["models"][0]["id"], "hidden-model");
    }
}
