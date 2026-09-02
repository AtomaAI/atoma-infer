//! The HTTP surface over the engine thread: the OpenAI-compatible chat completions endpoint, the
//! health and metrics endpoints an orchestrator polls, and the API docs.
//!
//! A request is tokenized on a blocking task, submitted through the engine's ingress, and read
//! back off its egress channel — whole for a completion, event by event for a stream. Overload is
//! a 429 and a gone engine a 503. Health is read from the engine thread's heartbeat, the free KV
//! block count is published as a gauge for whoever watches the pool, and the server stops when
//! the engine thread does, joining every executor thread and reporting why any of them stopped.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{anyhow, Context};
use atoma_core::engine::{Control, ControlSender, EngineHandle, EngineThread, IngressRefused};
use atoma_core::request::{egress, EgressReceiver, NewRequest, Priority, StopCriteria};
use atoma_core::types::TokenCount;
use atoma_engine::executor::ExecutorThread;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use metrics::gauge;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::json;
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::{signal, task, time};
use tracing::{error, info, instrument, warn, Span};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;
use uuid::Uuid;

use crate::api::chat_completions::{
    ChatCompletionResponse, CompletionIdentity, EngineRequest, Refused, RequestBody,
};
use crate::completion::{Completion, Failed, Progress};
use crate::detokenize::Detokenizer;
use crate::stream::Streamer;

/// The URL path to POST JSON for model chat completions.
pub const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";
/// The URL path reporting whether the node can still serve requests.
pub const HEALTHZ_PATH: &str = "/healthz";
/// The URL path rendering the recorded metrics, in the Prometheus text format.
pub const METRICS_PATH: &str = "/metrics";
/// The gauge carrying the engine's free KV block count.
pub const FREE_BLOCKS_GAUGE: &str = "atoma_engine_free_blocks";
/// How often the free-block gauge is sampled from the engine.
const GAUGE_INTERVAL: Duration = Duration::from_millis(500);
/// How long the shutdown waits for the executor threads to return: a follower caught inside a
/// collective when the leader died has no way back, and must not hold the process open.
const EXECUTOR_JOIN_TIMEOUT: Duration = Duration::from_secs(30);
/// How often the shutdown looks again at executor threads still running.
const EXECUTOR_JOIN_POLL: Duration = Duration::from_millis(50);

/// What every handler shares.
#[derive(Clone)]
pub struct AppState {
    pub engine: EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    /// The longest sequence the model serves: the bound on prompt plus completion.
    pub max_model_len: usize,
    /// How often a keep-alive comment is written to a stream with nothing to say.
    pub keep_alive: Duration,
    /// How old the heartbeat may be before the node reports itself unhealthy.
    pub heartbeat_stale_after: Duration,
}

/// The threads the server serves over, and joins when it stops.
pub struct EngineThreads {
    pub engine: EngineThread,
    pub executors: Vec<ExecutorThread>,
}

#[derive(OpenApi)]
#[openapi(
    paths(completion_handler),
    components(schemas(ChatCompletionResponse, RequestBody)),
    tags(
        (name = "Atoma's Chat Completions", description = "Atoma's Chat completion API")
    )
)]
pub struct ApiDoc;

/// Serves until Ctrl+C or the engine thread exiting, then shuts the engine down and joins every
/// thread.
///
/// # Errors
///
/// Returns an error when the server cannot start, and when any executor thread stopped on an
/// error: the cause a rank died of ends the server with it.
pub async fn run_server(
    listener: TcpListener,
    state: AppState,
    threads: EngineThreads,
) -> anyhow::Result<()> {
    // Installing the recorder is what makes the `metrics!` macros record anywhere but a no-op
    // sink, so it has to happen before the first request is served.
    let prometheus_handle = PrometheusBuilder::new()
        .install_recorder()
        .context("the Prometheus metrics recorder could not be installed")?;
    let router = build_router(state.clone(), prometheus_handle);
    let gauge_task = tokio::spawn(publish_free_blocks(state.engine.control.clone()));

    let EngineThreads { engine, executors } = threads;
    let (engine_exited, engine_exit) = oneshot::channel();
    let engine_join = task::spawn_blocking(move || {
        engine.join();
        // The server may already be shutting down for another reason and no longer listening.
        match engine_exited.send(()) {
            Ok(()) | Err(()) => {}
        }
    });
    let shutdown = async move {
        tokio::select! {
            _ = signal::ctrl_c() => info!("Ctrl+C; shutting down"),
            _ = engine_exit => error!("the engine thread exited; shutting down"),
        }
    };
    info!("OpenAI API server running");
    axum::serve(listener, router.into_make_service())
        .with_graceful_shutdown(shutdown)
        .await?;

    // The send waits for room in a full control channel rather than losing the shutdown, which
    // would leave the join below waiting forever; the engine drains control every pass.
    if state.engine.control.send(Control::Shutdown).is_err() {
        info!("the engine thread had already exited");
    }
    engine_join.await.context("the engine thread panicked")?;
    gauge_task.abort();
    join_executors(executors)
}

/// Joins every executor thread that returns within [`EXECUTOR_JOIN_TIMEOUT`], reporting the
/// first cause any of them stopped on, and giving up on any still running at the deadline.
fn join_executors(mut executors: Vec<ExecutorThread>) -> anyhow::Result<()> {
    let deadline = Instant::now() + EXECUTOR_JOIN_TIMEOUT;
    let mut failure = None;
    while !executors.is_empty() {
        let (finished, still_running): (Vec<_>, Vec<_>) =
            executors.into_iter().partition(ExecutorThread::is_finished);
        executors = still_running;
        for executor in finished {
            let rank = executor.rank();
            match executor.join() {
                Ok(()) => info!(%rank, "executor thread returned"),
                Err(cause) => {
                    error!(%rank, %cause, "executor thread failed");
                    failure.get_or_insert_with(|| anyhow!("executor rank {rank} failed: {cause}"));
                }
            }
        }
        if executors.is_empty() {
            break;
        }
        if Instant::now() >= deadline {
            let ranks: Vec<String> = executors.iter().map(|e| e.rank().to_string()).collect();
            error!(ranks = ?ranks, "executor threads did not return; giving up on them");
            failure.get_or_insert_with(|| {
                anyhow!(
                    "executor ranks {} did not return within {:?}",
                    ranks.join(", "),
                    EXECUTOR_JOIN_TIMEOUT
                )
            });
            break;
        }
        thread::sleep(EXECUTOR_JOIN_POLL);
    }
    failure.map_or(Ok(()), Err)
}

/// Publishes the engine's free block count every [`GAUGE_INTERVAL`], until the engine is gone.
async fn publish_free_blocks(control: ControlSender) {
    let mut ticks = time::interval(GAUGE_INTERVAL);
    loop {
        ticks.tick().await;
        if control.engine_gone() {
            return;
        }
        if let Some(free_blocks) = sample_free_blocks(&control).await {
            record_free_blocks(free_blocks);
        }
    }
}

/// Asks the engine for its state and returns its free block count, or nothing when the engine
/// did not take the query or answer it.
async fn sample_free_blocks(control: &ControlSender) -> Option<usize> {
    let (reply, answer) = flume::bounded(1);
    if control.try_send(Control::State { reply }).is_err() {
        return None;
    }
    answer
        .recv_async()
        .await
        .ok()
        .map(|state| state.free_blocks)
}

/// Records `free_blocks` on the gauge the pool's watchers read.
fn record_free_blocks(free_blocks: usize) {
    gauge!(FREE_BLOCKS_GAUGE).set(free_blocks as f64);
}

/// Builds the HTTP router the server serves: the OpenAI-compatible endpoint, the operational
/// endpoints an orchestrator polls, and the API docs.
pub fn build_router(state: AppState, prometheus_handle: PrometheusHandle) -> Router {
    Router::new()
        .route(CHAT_COMPLETIONS_PATH, post(completion_handler))
        .route(HEALTHZ_PATH, get(health_handler))
        .with_state(state)
        .merge(
            Router::new()
                .route(METRICS_PATH, get(metrics_handler))
                .with_state(prometheus_handle),
        )
        .merge(SwaggerUi::new("/docs").url("/api-docs/openapi.json", ApiDoc::openapi()))
}

/// Reports whether this node can still serve requests: the engine thread is there, has passed at
/// least once, and its heartbeat is recent. Liveness is read from the thread that could wedge,
/// not from this process being able to answer.
#[instrument(skip_all)]
async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    if state.engine.control.engine_gone() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "unavailable", "reason": "the engine thread has exited" })),
        );
    }
    let beat = state.engine.heartbeat.read();
    if beat.pass == 0 {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unavailable",
                "reason": "the engine thread has not completed a pass yet",
            })),
        );
    }
    let age = SystemTime::now()
        .duration_since(beat.at)
        .unwrap_or_default();
    if age > state.heartbeat_stale_after {
        warn!(
            age_millis = age.as_millis(),
            pass = beat.pass,
            "engine heartbeat is stale"
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unavailable",
                "reason": format!("the engine heartbeat is {} ms old", age.as_millis()),
            })),
        );
    }
    (
        StatusCode::OK,
        Json(json!({ "status": "ok", "pass": beat.pass })),
    )
}

/// Renders the metrics recorded so far, in the Prometheus text format.
#[instrument(skip_all)]
async fn metrics_handler(State(prometheus_handle): State<PrometheusHandle>) -> impl IntoResponse {
    prometheus_handle.render()
}

/// What a request that could not be served answers with.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
    request_id: String,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, message: String, request_id: &str) -> Self {
        Self {
            status,
            kind,
            message,
            request_id: request_id.to_owned(),
        }
    }

    fn refused(refused: &Refused, request_id: &str) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            refused.to_string(),
            request_id,
        )
    }

    fn failed(failed: &Failed, request_id: &str) -> Self {
        Self::new(
            failed.status(),
            failed.kind(),
            failed.to_string(),
            request_id,
        )
    }

    fn internal(message: String, request_id: &str) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            message,
            request_id,
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "request_id": self.request_id,
            }
        }));
        (self.status, body).into_response()
    }
}

/// The id an arriving request is known by, here and everywhere downstream.
///
/// UUIDv7: the leading 48 bits are the arrival time in milliseconds, so an id carries its own
/// timestamp into the engine's logs and ids issued in different milliseconds sort by arrival. The
/// rest is random, which keeps ids unique across processes and across restarts.
fn next_request_id() -> Uuid {
    Uuid::now_v7()
}

/// Seconds since the Unix epoch, as the API reports a completion's creation.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// Serves one chat completion request: whole, or as a stream of chunks.
#[utoipa::path(
    post,
    path = CHAT_COMPLETIONS_PATH,
    request_body = RequestBody,
    responses(
        (status = 200, description = "The completion, whole or streamed", body = ChatCompletionResponse),
        (status = 400, description = "Asks for what the engine cannot honour", body = serde_json::Value),
        (status = 429, description = "The engine is overloaded", body = serde_json::Value),
        (status = 503, description = "The engine is gone", body = serde_json::Value)
    ),
)]
#[instrument(skip_all, fields(request_id))]
pub async fn completion_handler(
    State(state): State<AppState>,
    Json(request): Json<RequestBody>,
) -> Result<Response, ApiError> {
    let request_id = next_request_id().to_string();
    Span::current().record("request_id", request_id.as_str());
    let model = request.model().to_string();
    let engine_request = request
        .to_engine_request(rand::random())
        .map_err(|refused| ApiError::refused(&refused, &request_id))?;
    let stream = engine_request.stream;
    let (receiver, completion) = submit(&state, &request_id, model, engine_request).await?;
    if stream {
        let streamer = Streamer::new(receiver, completion);
        let keep_alive = KeepAlive::new()
            .interval(state.keep_alive)
            .text("keep-alive");
        return Ok(Sse::new(streamer).keep_alive(keep_alive).into_response());
    }
    let completion = collect(receiver, completion, &request_id).await?;
    Ok(Json(completion).into_response())
}

/// Tokenizes the prompt on a blocking task and submits the request to the engine, handing back
/// its egress and the completion that reads it.
async fn submit(
    state: &AppState,
    request_id: &str,
    model: String,
    request: EngineRequest,
) -> Result<(EgressReceiver, Completion), ApiError> {
    let EngineRequest {
        prompt,
        sampling,
        max_new_tokens,
        stop,
        stream: _,
    } = request;
    let tokenizer = Arc::clone(&state.tokenizer);
    let prompt_ids = task::spawn_blocking(move || {
        tokenizer
            .encode(prompt, true)
            .map(|encoding| encoding.get_ids().to_vec())
    })
    .await
    .map_err(|join| ApiError::internal(format!("tokenization panicked: {join}"), request_id))?
    .map_err(|error| {
        ApiError::internal(
            format!("the prompt cannot be tokenized: {error}"),
            request_id,
        )
    })?;
    let prompt_tokens = prompt_ids.len();
    let max_new_tokens = match max_new_tokens {
        Some(budget) => budget,
        None => {
            TokenCount::new(state.max_model_len.saturating_sub(prompt_tokens)).ok_or_else(|| {
                ApiError::failed(
                    &Failed::PromptTooLong {
                        prompt_tokens,
                        max_model_length: state.max_model_len,
                    },
                    request_id,
                )
            })?
        }
    };
    let (sender, receiver) = egress();
    let new_request = NewRequest {
        prompt: prompt_ids,
        sampling,
        stop: StopCriteria {
            max_new_tokens,
            ignore_eos: false,
        },
        priority: Priority::default(),
        egress: sender,
    };
    match state.engine.ingress.try_send(new_request) {
        Ok(()) => {}
        Err(IngressRefused::Overload(_)) => {
            return Err(ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "overloaded",
                "the engine cannot take another request right now".to_owned(),
                request_id,
            ));
        }
        Err(IngressRefused::EngineGone(_)) => {
            return Err(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "unavailable",
                "the engine is gone".to_owned(),
                request_id,
            ));
        }
    }
    let detokenizer = Detokenizer::new(Arc::clone(&state.tokenizer), stop);
    let identity = CompletionIdentity {
        id: request_id.to_owned(),
        model,
        created: unix_now(),
    };
    Ok((
        receiver,
        Completion::new(identity, detokenizer, prompt_tokens),
    ))
}

/// Reads a request's events until it finishes and answers with the whole completion. Dropping
/// the receiver on the way out is the cancel a stop string needs.
async fn collect(
    receiver: EgressReceiver,
    mut completion: Completion,
    request_id: &str,
) -> Result<ChatCompletionResponse, ApiError> {
    loop {
        let Ok(event) = receiver.recv_async().await else {
            return Err(ApiError::internal(
                "the engine closed the request without a finish".to_owned(),
                request_id,
            ));
        };
        let progress = completion.apply(event).map_err(|error| {
            ApiError::internal(format!("a token cannot be decoded: {error}"), request_id)
        })?;
        match progress {
            Progress::Nothing | Progress::Text(_) => {}
            Progress::Finished {
                finish_reason,
                usage,
                ..
            } => return Ok(completion.response(finish_reason, usage)),
            Progress::Failed(failed) => return Err(ApiError::failed(&failed, request_id)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::convert::Infallible;
    use std::time::Duration;

    use atoma_core::attention::{
        BackendDeclaration, CaptureContract, ModelDeclaration, SupportLevel,
    };
    use atoma_core::dispatch::{BucketLadder, DispatchConfig};
    use atoma_core::engine::{control, heartbeat, ingress, Engine, EngineConfig, EngineThread};
    use atoma_core::kv::HashAlgorithm;
    use atoma_core::scheduler::{AdmissionPolicy, SchedulerConfig};
    use atoma_core::step::StepCommand;
    use atoma_core::types::{RequestCount, TokenCount};
    use atoma_engine::batch::BatchLayout;
    use atoma_engine::executor::{Executor, ExecutorError, ExecutorLoop};
    use atoma_engine::forward::Forward;
    use atoma_engine::logits::Logits;
    use axum::body::{to_bytes, Body};
    use axum::http::{header, Method, Request};
    use crossbeam_utils::sync::Parker;
    use metrics_exporter_prometheus::PrometheusBuilder;
    use serde_json::{json, Value};
    use tokenizers::Tokenizer;
    use tower::ServiceExt;

    use super::*;
    use crate::test_support::tokenizer;

    const MAX_MODEL_LEN: usize = 512;
    const BLOCK_SIZE: usize = 16;
    const MAX_BATCH: usize = 4;
    /// How long a test waits on the engine thread before calling it wedged.
    const WAIT: Duration = Duration::from_secs(30);

    /// A forward whose every selected row peaks at one token, so greedy sampling returns it.
    struct ConstantForward {
        token: u32,
        vocab: usize,
        logits: Vec<f32>,
    }

    impl Forward for ConstantForward {
        type Error = Infallible;

        fn forward(
            &mut self,
            _command: &StepCommand,
            layout: &BatchLayout,
        ) -> Result<Logits<'_>, Infallible> {
            let rows = layout.selected.len();
            self.logits.clear();
            self.logits.resize(rows * self.vocab, 0.0);
            for row in 0..rows {
                self.logits[row * self.vocab + self.token as usize] = 1.0;
            }
            Ok(Logits::new(&self.logits, self.vocab))
        }
    }

    fn engine_config() -> EngineConfig {
        let nonzero = |value: usize| TokenCount::new(value).unwrap();
        let requests = |value: usize| RequestCount::new(value).unwrap();
        let slots = 8;
        EngineConfig {
            scheduler: SchedulerConfig {
                token_budget: nonzero(2048),
                max_batch: requests(MAX_BATCH),
                max_model_len: nonzero(MAX_MODEL_LEN),
                block_size: nonzero(BLOCK_SIZE),
                window: requests(8),
                admission: AdmissionPolicy::Fcfs,
                max_requests: requests(slots),
                max_client_backlog: nonzero(1024),
                eos_token_ids: Vec::new(),
                hash_algorithm: HashAlgorithm::Sha256V1,
            },
            dispatch: DispatchConfig {
                bucket_ladder: BucketLadder::new(vec![1, 2, 4]).unwrap(),
                captured_max_requests: requests(MAX_BATCH),
            },
            block_count: u32::try_from(slots * MAX_MODEL_LEN / BLOCK_SIZE + MAX_BATCH).unwrap(),
            ingress_capacity: requests(8),
            idle_deadline: Duration::from_millis(1),
        }
    }

    /// A running engine with an executor over a constant forward on a thread of its own, and
    /// the state the router serves it through.
    struct Served {
        state: AppState,
        engine: EngineThread,
        executor: thread::JoinHandle<Result<(), ExecutorError>>,
        tokenizer: Arc<Tokenizer>,
    }

    fn serve(token: &str, heartbeat_stale_after: Duration) -> Served {
        let tokenizer = tokenizer();
        let token_id = tokenizer.token_to_id(token).expect("a vocabulary token");
        let vocab = tokenizer.get_vocab_size(true);
        let contract = CaptureContract::resolve(
            &[BackendDeclaration::new("test", SupportLevel::Never)],
            &ModelDeclaration::new("test"),
        );
        let (handle, rings, engine) = Engine::spawn(&engine_config(), &contract).unwrap();
        let forward = ConstantForward {
            token: token_id,
            vocab,
            logits: Vec::new(),
        };
        let executor = Executor::new(rings, forward, TokenCount::new(BLOCK_SIZE).unwrap());
        let executor = thread::spawn(move || executor.run());
        // Health reads the heartbeat, which the thread publishes after its first pass.
        let started = Instant::now();
        while handle.heartbeat.read().pass == 0 {
            assert!(started.elapsed() < WAIT, "the engine thread never passed");
            thread::sleep(Duration::from_millis(1));
        }
        let state = AppState {
            engine: handle,
            tokenizer: Arc::clone(&tokenizer),
            max_model_len: MAX_MODEL_LEN,
            keep_alive: Duration::from_millis(100),
            heartbeat_stale_after,
        };
        Served {
            state,
            engine,
            executor,
            tokenizer,
        }
    }

    impl Served {
        fn router(&self) -> Router {
            build_router(
                self.state.clone(),
                PrometheusBuilder::new().build_recorder().handle(),
            )
        }

        fn shutdown(self) {
            self.state
                .engine
                .control
                .try_send(Control::Shutdown)
                .unwrap();
            self.engine.join();
            self.executor.join().unwrap().unwrap();
        }
    }

    async fn post(router: Router, body: Value) -> (StatusCode, String) {
        let response = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(CHAT_COMPLETIONS_PATH)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    async fn get(router: Router, path: &str) -> (StatusCode, String) {
        let response = router
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    fn completion_body(fields: Value) -> Value {
        let mut body = json!({
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "messages": [{ "role": "user", "content": "Hi" }],
            "max_completion_tokens": 4,
            "temperature": 0.0
        });
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        body
    }

    #[tokio::test]
    async fn a_completion_is_served_whole_through_the_engine_and_the_executor() {
        let served = serve("a", Duration::from_secs(30));
        let (status, body) = post(served.router(), completion_body(json!({}))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "meta-llama/Llama-3.2-1B-Instruct");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "aaaa");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["completion_tokens"], 4);
        let prompt_tokens = served
            .tokenizer
            .encode("<|begin_of_text|>", true)
            .unwrap()
            .get_ids()
            .len();
        assert!(
            body["usage"]["prompt_tokens"].as_u64().unwrap() as usize > prompt_tokens,
            "the prompt is the templated conversation: {body}"
        );
        served.shutdown();
    }

    #[tokio::test]
    async fn a_completion_is_streamed_as_chunks_then_the_finish_then_done() {
        let served = serve("a", Duration::from_secs(30));
        let (status, body) =
            post(served.router(), completion_body(json!({ "stream": true }))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let events: Vec<Value> = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|payload| *payload != "[DONE]")
            .map(|payload| serde_json::from_str(payload).unwrap())
            .collect();
        assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
        let text: String = events
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, "aaaa");
        let last = events.last().unwrap();
        assert_eq!(last["object"], "chat.completion.chunk");
        assert_eq!(last["choices"][0]["finish_reason"], "length");
        assert_eq!(last["usage"]["completion_tokens"], 4);
        assert!(
            events[..events.len() - 1]
                .iter()
                .all(|chunk| chunk["choices"][0].get("finish_reason").is_none()),
            "only the last chunk carries the finish: {events:?}"
        );
        served.shutdown();
    }

    #[tokio::test]
    async fn a_stop_string_ends_the_completion_before_the_match() {
        let served = serve("a", Duration::from_secs(30));
        let (status, body) = post(
            served.router(),
            completion_body(json!({ "stop": "aa", "max_completion_tokens": 8 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], "");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert!(
            body["usage"]["completion_tokens"].as_u64().unwrap() < 8,
            "cancelled at the match, not run to the budget: {body}"
        );
        served.shutdown();
    }

    #[tokio::test]
    async fn what_the_engine_cannot_honour_is_a_400() {
        let served = serve("a", Duration::from_secs(30));
        let (status, body) = post(served.router(), completion_body(json!({ "n": 2 }))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("2 choices"));
        served.shutdown();
    }

    #[tokio::test]
    async fn a_prompt_with_no_room_left_is_a_400() {
        let served = serve("a", Duration::from_secs(30));
        let long = "word ".repeat(MAX_MODEL_LEN);
        let (status, body) = post(
            served.router(),
            json!({
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": [{ "role": "user", "content": long }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body.contains("max model length"), "{body}");
        served.shutdown();
    }

    #[tokio::test]
    async fn a_gone_engine_is_a_503_and_unhealthy() {
        let served = serve("a", Duration::from_secs(30));
        let router = served.router();
        let (status, body) = get(router.clone(), HEALTHZ_PATH).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        served.shutdown();

        let (status, body) = post(router.clone(), completion_body(json!({}))).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
        assert!(body.contains("unavailable"), "{body}");
        let (status, body) = get(router, HEALTHZ_PATH).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
        assert!(body.contains("exited"), "{body}");
    }

    /// An engine thread that is there but never passes again: its channels stay open and its
    /// heartbeat stops. Health must read the beat, not the thread being there.
    #[tokio::test]
    async fn a_wedged_engine_is_unhealthy_by_its_heartbeat_age() {
        let parker = Parker::new();
        let (ingress_sender, _ingress) = ingress(8, parker.unparker().clone());
        let (control_sender, _control) = control(parker.unparker().clone());
        let (publisher, reader) = heartbeat();
        publisher.publish(1);
        let state = AppState {
            engine: EngineHandle {
                ingress: ingress_sender,
                control: control_sender,
                heartbeat: reader,
            },
            tokenizer: tokenizer(),
            max_model_len: MAX_MODEL_LEN,
            keep_alive: Duration::from_millis(100),
            heartbeat_stale_after: Duration::from_millis(5),
        };
        let router = build_router(state, PrometheusBuilder::new().build_recorder().handle());
        time::sleep(Duration::from_millis(20)).await;
        let (status, body) = get(router, HEALTHZ_PATH).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
        assert!(body.contains("ms old"), "{body}");
    }

    #[tokio::test]
    async fn the_free_block_gauge_is_sampled_from_the_engine_and_recorded() {
        let served = serve("a", Duration::from_secs(30));
        let free_blocks = sample_free_blocks(&served.state.engine.control)
            .await
            .expect("the engine answers a state query");
        assert!(free_blocks > 0);
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || record_free_blocks(free_blocks));
        let router = build_router(served.state.clone(), handle);
        let (status, body) = get(router, METRICS_PATH).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body.contains(&format!("{FREE_BLOCKS_GAUGE} {free_blocks}")),
            "{body}"
        );
        served.shutdown();
        assert_eq!(
            sample_free_blocks(&served_control_after_shutdown()).await,
            None,
            "a gone engine answers nothing"
        );
    }

    /// A control sender whose engine has exited.
    fn served_control_after_shutdown() -> ControlSender {
        let parker = Parker::new();
        let (sender, receiver) = control(parker.unparker().clone());
        drop(receiver);
        sender
    }

    #[tokio::test]
    async fn chat_completions_is_mounted_as_a_post_route() {
        let served = serve("a", Duration::from_secs(30));
        let (status, _) = get(served.router(), CHAT_COMPLETIONS_PATH).await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
        let (status, _) = get(served.router(), "/not-a-route").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        served.shutdown();
    }

    /// Two requests must never share an id: ids key the engine's client channels, and a
    /// collision hands one client another's tokens.
    #[test]
    fn request_ids_are_unique() {
        const NUM_IDS: usize = 1024;
        let ids = (0..NUM_IDS)
            .map(|_| next_request_id())
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), NUM_IDS);
    }

    #[test]
    fn request_ids_carry_their_arrival_time() {
        let before = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("Time went backwards");
        let (seconds, _) = next_request_id()
            .get_timestamp()
            .expect("A request id must carry a timestamp")
            .to_unix();
        assert!(
            seconds >= before.as_secs() && seconds <= before.as_secs() + 1,
            "id timestamp {seconds} is not the arrival time {}",
            before.as_secs()
        );
    }

    /// Ids issued in different milliseconds sort by arrival, which is what makes them useful for
    /// ordering a log. Within one millisecond UUIDv7 makes no such promise.
    #[test]
    fn request_ids_sort_by_arrival() {
        let earlier = next_request_id().to_string();
        thread::sleep(Duration::from_millis(2));
        let later = next_request_id().to_string();
        assert!(earlier < later, "{earlier} should sort before {later}");
    }
}
