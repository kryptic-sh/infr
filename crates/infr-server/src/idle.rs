use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};

use axum::{body::Body, extract::Request, middleware::Next, response::Response, Extension};
use http_body::{Body as HttpBody, Frame, SizeHint};
use tokio::{sync::Notify, time::Instant};

#[derive(Debug)]
pub(crate) struct Activity {
    state: Mutex<State>,
    changed: Notify,
}

#[derive(Debug)]
struct State {
    busy: usize,
    idle_since: Instant,
    stopping: bool,
}

impl Default for Activity {
    fn default() -> Self {
        Self {
            state: Mutex::new(State {
                busy: 0,
                idle_since: Instant::now(),
                stopping: false,
            }),
            changed: Notify::new(),
        }
    }
}

impl Activity {
    pub(crate) fn start_serving(&self) {
        let mut state = self.state.lock().unwrap();
        if state.busy == 0 {
            state.idle_since = Instant::now();
        }
        self.changed.notify_one();
    }

    #[cfg(test)]
    pub(crate) fn busy(&self) -> usize {
        self.state.lock().unwrap().busy
    }

    pub(crate) fn enter(self: &Arc<Self>) -> Option<Guard> {
        let mut state = self.state.lock().unwrap();
        if state.stopping {
            return None;
        }
        state.busy += 1;
        self.changed.notify_one();
        Some(Guard {
            _lease: Arc::new(Lease(self.clone())),
        })
    }

    pub(crate) async fn wait_idle(&self, timeout: Duration) {
        loop {
            let changed = self.changed.notified();
            let deadline = {
                let mut state = self.state.lock().unwrap();
                if state.busy == 0 && state.idle_since.elapsed() >= timeout {
                    state.stopping = true;
                    return;
                }
                if state.busy == 0 {
                    state.idle_since.checked_add(timeout)
                } else {
                    None
                }
            };
            match deadline {
                Some(deadline) => {
                    tokio::select! {
                        _ = tokio::time::sleep_until(deadline) => {}
                        _ = changed => {}
                    }
                }
                None => changed.await,
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Guard {
    _lease: Arc<Lease>,
}

#[derive(Debug)]
struct Lease(Arc<Activity>);

impl Drop for Lease {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap();
        state.busy -= 1;
        if state.busy == 0 {
            state.idle_since = Instant::now();
        }
        self.0.changed.notify_one();
    }
}

pub(crate) async fn track(
    Extension(activity): Extension<Arc<Activity>>,
    mut request: Request,
    next: Next,
) -> Response {
    let Some(guard) = activity.enter() else {
        return super::json_error(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down; retry on a new connection".into(),
        );
    };
    request.extensions_mut().insert(guard.clone());
    let response = next.run(request).await;
    response.map(|body| {
        let guard = (!body.is_end_stream()).then_some(guard);
        Body::new(TrackedBody { body, guard })
    })
}

struct TrackedBody {
    body: Body,
    guard: Option<Guard>,
}

impl HttpBody for TrackedBody {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let frame = Pin::new(&mut self.body).poll_frame(cx);
        if matches!(frame, Poll::Ready(None) | Poll::Ready(Some(Err(_))))
            || self.body.is_end_stream()
        {
            self.guard.take();
        }
        frame
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{http::Request, routing::get, Router};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[tokio::test(start_paused = true)]
    async fn idle_exit_closes_admission() {
        let activity = Arc::new(Activity::default());
        let waiter = tokio::spawn({
            let activity = activity.clone();
            async move { activity.wait_idle(Duration::from_secs(5)).await }
        });
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(4)).await;
        assert!(!waiter.is_finished());
        tokio::time::advance(Duration::from_secs(1)).await;
        waiter.await.unwrap();
        assert!(activity.enter().is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn last_guard_drop_starts_a_full_idle_interval() {
        let activity = Arc::new(Activity::default());
        let guard = activity.enter().unwrap();
        let clone = guard.clone();
        let waiter = tokio::spawn({
            let activity = activity.clone();
            async move { activity.wait_idle(Duration::from_secs(5)).await }
        });
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(50)).await;
        drop(guard);
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        drop(clone);
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(4)).await;
        assert!(!waiter.is_finished());
        tokio::time::advance(Duration::from_secs(1)).await;
        waiter.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn overlapping_requests_do_not_start_idle_early() {
        let activity = Arc::new(Activity::default());
        let first = activity.enter().unwrap();
        let second = activity.enter().unwrap();
        drop(first);
        tokio::time::advance(Duration::from_secs(50)).await;
        assert_eq!(activity.state.lock().unwrap().busy, 1);
        drop(second);
        assert_eq!(activity.state.lock().unwrap().idle_since, Instant::now());
    }

    #[tokio::test]
    async fn response_body_retains_activity_and_trailers() {
        let activity = Arc::new(Activity::default());
        let mut trailers = axum::http::HeaderMap::new();
        trailers.insert("x-finished", "yes".parse().unwrap());
        let router = Router::new()
            .route(
                "/",
                get(move || async move {
                    Body::new(http_body_util::StreamBody::new(futures_util::stream::iter(
                        [
                            Ok::<_, std::convert::Infallible>(Frame::data(
                                axum::body::Bytes::from_static(b"data"),
                            )),
                            Ok(Frame::trailers(trailers)),
                        ],
                    )))
                }),
            )
            .layer(axum::middleware::from_fn(track))
            .layer(Extension(activity.clone()));
        let response = router.oneshot(Request::new(Body::empty())).await.unwrap();
        assert_eq!(activity.state.lock().unwrap().busy, 1);
        let collected = response.into_body().collect().await.unwrap();
        assert_eq!(collected.trailers().unwrap()["x-finished"], "yes");
        assert_eq!(activity.state.lock().unwrap().busy, 0);
    }

    #[tokio::test]
    async fn empty_or_dropped_response_releases_activity() {
        let activity = Arc::new(Activity::default());
        let router = Router::new()
            .route("/", get(|| async { "hello" }))
            .route("/empty", get(|| async { axum::http::StatusCode::OK }))
            .layer(axum::middleware::from_fn(track))
            .layer(Extension(activity.clone()));
        let response = router
            .clone()
            .oneshot(Request::new(Body::empty()))
            .await
            .unwrap();
        assert_eq!(activity.state.lock().unwrap().busy, 1);
        drop(response);
        assert_eq!(activity.state.lock().unwrap().busy, 0);
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/empty")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(activity.state.lock().unwrap().busy, 0);
        drop(response);
    }
}
