use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use atoma_backends::{GenerateRequest, ServiceRequest};
use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{
        sse::{KeepAlive, KeepAliveStream},
        IntoResponse, Sse,
    },
    routing::{get, post},
    Json, Router,
};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::{json, Value};
use tokio::{
    net::TcpListener,
    signal,
    sync::{
        mpsc::{self, UnboundedSender},
        oneshot,
    },
    task::JoinHandle,
};
use tracing::{error, info, instrument};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::{
    api::chat_completions::{ChatCompletionResponse, RequestBody},
    stream::Streamer,
};

/// The URL path to POST JSON for model chat completions.
pub const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";
/// The URL path reporting whether the node can still serve requests.
pub const HEALTHZ_PATH: &str = "/healthz";
/// The URL path rendering the recorded metrics, in the Prometheus text format.
pub const METRICS_PATH: &str = "/metrics";
pub const AUTH_BEARER_PREFIX: &str = "Bearer ";

/// Represents the shared state of the application.
///
/// This struct contains various components that are shared across different parts of the
/// application, particularly for handling requests and managing the lifecycle of the server.
#[derive(Clone)]
pub struct AppState {
    /// A counter for generating unique request IDs.
    ///
    /// This atomic counter is incremented for each new request, ensuring unique identification.
    pub request_counter: Arc<AtomicU64>,
    /// A sender for non-streaming LLM service requests.
    ///
    /// This channel is used to send generate requests to the LLM service and receive
    /// the output through a oneshot channel.
    pub llm_service_sender: UnboundedSender<ServiceRequest>,
    /// A sender for the shutdown signal.
    ///
    /// This channel is used to send a shutdown signal to gracefully stop the server.
    pub shutdown_signal_sender: mpsc::Sender<()>,
    /// The interval at which the server sends keep-alive messages to the client during streaming.
    ///
    /// This value represents the number of milliseconds between each keep-alive message sent
    /// to the client during a streaming response. Keep-alive messages help maintain the connection
    /// and prevent timeouts, especially for long-running requests.
    ///
    /// A lower value will send keep-alive messages more frequently, which can be useful for
    /// unstable connections but may increase network traffic. A higher value reduces network
    /// overhead but might risk connection timeouts on less stable networks.
    ///
    /// Typical values range from 1000 (1 second) to 30000 (30 seconds), depending on the specific
    /// requirements of the application and the expected network conditions of clients.    
    pub streaming_interval_in_millis: u64,
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

/// Runs the Axums server and manages its lifecycle, including graceful shutdown.
///
/// This function sets up the HTTP router, starts the server, and handles the shutdown process
/// for both the server and the LLM service.
///
/// # Arguments
///
/// * `listener` - A `TcpListener` that the server will bind to.
/// * `app_state` - The `AppState` containing shared application data.
/// * `join_handle` - A `JoinHandle` for the LLM service task.
///
/// # Returns
///
/// Returns `Ok(())` if the server runs and shuts down successfully, or an error if any part
/// of the process fails.
///
/// # Flow
///
/// 1. Sets up the HTTP router with the chat completions route.
/// 2. Configures a shutdown signal to listen for Ctrl+C.
/// 3. Starts the server with graceful shutdown capabilities.
/// 4. Waits for the shutdown signal.
/// 5. Initiates shutdown of the app state and LLM service.
/// 6. Waits for the LLM service to shut down, with a timeout.
///
/// # Error Handling
///
/// - Returns an error if the server fails to start or encounters an error during operation.
/// - Logs errors if the LLM service encounters issues during shutdown or times out.
pub async fn run_server(
    listener: TcpListener,
    app_state: AppState,
    join_handle: JoinHandle<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let shutdown_signal_sender = app_state.shutdown_signal_sender.clone();
    // Installing the recorder is what makes the `metrics!` macros throughout the engine record
    // anywhere but a no-op sink, so it has to happen before the first request is served.
    let prometheus_handle = PrometheusBuilder::new()
        .install_recorder()
        .context("Failed to install the Prometheus metrics recorder")?;
    let http_router = build_router(app_state, prometheus_handle);

    let shutdown_signal = async {
        signal::ctrl_c()
            .await
            .expect("Failed to parse Ctrl+C signal");
        info!("Shutting down server...");
    };

    let server = axum::serve(listener, http_router.into_make_service())
        .with_graceful_shutdown(shutdown_signal);

    info!("OpenAI API server running, press Ctrl+C to shut it down");
    server.await?;

    // Shutdown the app state
    shutdown_signal_sender.send(()).await?;

    // Wait for LLM service to gracefully shut down
    match tokio::time::timeout(Duration::from_secs(30), join_handle).await {
        Ok(Ok(_)) => {
            info!("LlmService shutdown successfully");
        }
        Ok(Err(e)) => {
            error!("LlmService encountered an error during shutdown: {:?}", e);
        }
        Err(_) => {
            error!("LlmService shutdown timed out");
        }
    }
    info!("Server and LlmService shutdown complete");

    Ok(())
}

/// Builds the HTTP router the server serves: the OpenAI-compatible endpoint, the operational
/// endpoints an orchestrator polls, and the API docs.
pub fn build_router(app_state: AppState, prometheus_handle: PrometheusHandle) -> Router {
    Router::new()
        .route(CHAT_COMPLETIONS_PATH, post(completion_handler))
        .route(HEALTHZ_PATH, get(health_handler))
        .with_state(app_state)
        .merge(
            Router::new()
                .route(METRICS_PATH, get(metrics_handler))
                .with_state(prometheus_handle),
        )
        .merge(SwaggerUi::new("/docs").url("/api-docs/openapi.json", ApiDoc::openapi()))
}

/// Reports whether this node can still accept requests.
///
/// The check is the engine channel: once the `LlmService` is gone, its receiver is dropped and no
/// request submitted from here can ever be answered, so the node is unhealthy rather than merely
/// idle.
#[instrument(skip_all)]
async fn health_handler(app_state: State<AppState>) -> impl IntoResponse {
    if app_state.llm_service_sender.is_closed() {
        error!("The LLM service is no longer accepting requests");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "unavailable", "reason": "llm service is not running" })),
        );
    }

    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

/// Renders the metrics recorded so far, in the Prometheus text format.
#[instrument(skip_all)]
async fn metrics_handler(prometheus_handle: State<PrometheusHandle>) -> impl IntoResponse {
    prometheus_handle.render()
}

/// Represents the response from a chat completion request.
/// This enum can either be a full completion response or a stream chunk.
#[derive(Debug)]
pub enum ChatResponse {
    /// A complete chat completion response.
    Completion(ChatCompletionResponse),
    /// A chunk of a streaming chat completion response.
    Stream(Sse<KeepAliveStream<Streamer>>),
}

impl IntoResponse for ChatResponse {
    fn into_response(self) -> axum::response::Response {
        match self {
            ChatResponse::Completion(completion) => Json(completion).into_response(),
            ChatResponse::Stream(stream) => stream.into_response(),
        }
    }
}

/// Handles chat completion requests by processing the input, sending it to the LLM service,
/// and returning the generated response.
///
/// # Arguments
///
/// * `app_state` - The shared application state containing the request counter and LLM service
///   sender.
/// * `headers` - The HTTP headers of the incoming request.
/// * `request` - The deserialized JSON request body containing the chat completion parameters.
///
/// # Returns
///
/// Returns a `Result` containing either:
/// - `Ok(Json<ChatCompletionResponse>)`: The successful chat completion response.
/// - `Err((StatusCode, Json<serde_json::Value>))`: An error response with appropriate status code
///   and message.
///
/// # Flow
///
/// 1. Generates a unique request ID.
/// 2. Extracts the authorization key from headers (currently unused).
/// 3. Converts the request to a `GenerateRequest` for the LLM service.
/// 4. Sends the request to the LLM service and awaits the response.
/// 5. Converts the LLM service output to a `ChatCompletionResponse`.
/// 6. Returns the response or an error if any step fails.
///
/// # Error Handling
///
/// - Returns a 500 Internal Server Error if:
///   - Sending the request to the LLM service fails.
///   - Receiving the response from the LLM service fails.
///   - Converting the LLM output to a `ChatCompletionResponse` fails.
#[utoipa::path(
    post,
    path = CHAT_COMPLETIONS_PATH,
    request_body = RequestBody,
    responses(
        (status = 200, description = "Successful chat completion response", body = ChatCompletionResponse),
        (status = 500, description = "Internal server error", body = serde_json::Value)
    ),
)]
#[instrument(skip_all, fields(request_id))]
pub async fn completion_handler(
    app_state: State<AppState>,
    headers: HeaderMap,
    Json(request): Json<RequestBody>,
) -> Result<ChatResponse, (StatusCode, Json<serde_json::Value>)> {
    let request_number = app_state.request_counter.fetch_add(1, Ordering::SeqCst);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_nanos();
    let request_id = format!("{request_number}-{now}");

    let _auth_key = headers
        .get(header::AUTHORIZATION)
        .and_then(|auth_value| -> Option<&str> {
            auth_value
                .to_str()
                .ok()
                .and_then(|auth_header_str| auth_header_str.strip_prefix(AUTH_BEARER_PREFIX))
        })
        .unwrap_or_default();
    let model = request.model().to_string();
    let stream = request.stream().unwrap_or(false);
    let generate_request = request.to_generate_request(request_id.clone());

    let chat_response = if stream {
        ChatResponse::Stream(
            handle_generate_stream_request(&app_state, model, generate_request).await?,
        )
    } else {
        ChatResponse::Completion(
            handle_generate_request(&app_state, model, generate_request).await?,
        )
    };

    Ok(chat_response)
}

/// Handles a generate request by sending it to the LLM service and processing the response.
///
/// This function is responsible for:
/// 1. Sending the generate request to the LLM service
/// 2. Awaiting and processing the response
/// 3. Converting the output to a ChatCompletionResponse
/// 4. Handling any errors that occur during this process
///
/// # Arguments
///
/// * `app_state` - A reference to the application state, which contains the LLM service sender.
/// * `generate_request` - The GenerateRequest to be sent to the LLM service.
///
/// # Returns
///
/// Returns a `Result` containing either:
/// - `Ok(Json<ChatCompletionResponse>)`: The successful chat completion response.
/// - `Err((StatusCode, Json<Value>))`: An error response with appropriate status code and message.
///
/// # Errors
///
/// This function will return an error in the following cases:
/// - If sending the request to the LLM service fails
/// - If receiving the response from the LLM service fails
/// - If converting the LLM output to a ChatCompletionResponse fails
///
/// In all error cases, a 500 Internal Server Error is returned with a JSON body
/// containing an error message and type.
///
/// # Note
///
/// The `request_id` and `model` variables are used in the error responses but are not
/// defined in the current function signature. These should be added as parameters
/// or derived from the `generate_request` to ensure all error responses include
/// the correct request ID and model information.
#[instrument(skip_all)]
async fn handle_generate_request(
    app_state: &AppState,
    model: String,
    generate_request: GenerateRequest,
) -> Result<ChatCompletionResponse, (StatusCode, Json<Value>)> {
    let request_id = generate_request.request_id.clone();
    let (sender, receiver) = oneshot::channel();
    if let Err(send_error) = app_state
        .llm_service_sender
        .send(ServiceRequest::GenerateRequest(generate_request, sender))
    {
        error!("Failed to send request to LLM Service: {}", send_error);
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Internal server error: failed to process request",
                    "type": "internal_error",
                    "request_id": request_id,
                }
            })),
        ));
    }
    let outputs = match receiver.await {
        Ok(outputs) => outputs,
        Err(_) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": "Failed to receive response from LLM Service",
                        "type": "internal_error",
                        "request_id": request_id,
                    }
                })),
            ));
        }
    };

    let chat_response = ChatCompletionResponse::try_from((model, outputs)).map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": err,
                    "type": "internal_error",
                    "request_id": request_id,
                }
            })),
        )
    })?;

    Ok(chat_response)
}

/// Handles a streaming generate request by sending it to the LLM service and processing the
/// response.
///
/// This function is responsible for:
/// 1. Sending the generate stream request to the LLM service
/// 2. Setting up a streaming response channel
/// 3. Creating a Streamer to manage the response stream
/// 4. Configuring Server-Sent Events (SSE) with keep-alive functionality
///
/// # Arguments
///
/// * `app_state` - A reference to the `AppState` containing shared application data.
/// * `generate_request` - The `GenerateRequest` to be sent to the LLM service.
///
/// # Returns
///
/// Returns a `Result` containing either:
/// - `Ok(impl IntoResponse)`: A streaming response wrapped in an `Sse` struct.
/// - `Err((StatusCode, Json<Value>))`: An error response with status code and JSON body.
///
/// # Error Handling
///
/// If sending the request to the LLM service fails, this function returns a 500 Internal
/// Server Error with a JSON body containing an error message and type.
///
/// # Streaming Behavior
///
/// The function sets up a streaming channel using `flume::unbounded()` and creates a `Streamer`
/// to manage the response stream. The `Sse` struct is used to wrap the streamer and configure
/// Server-Sent Events, including keep-alive functionality to maintain the connection.
///
/// # Keep-Alive Configuration
///
/// Keep-alive messages are sent at intervals specified by `app_state.streaming_interval_in_millis`.
/// These messages help prevent connection timeouts during long-running requests.
#[instrument(skip_all)]
async fn handle_generate_stream_request(
    app_state: &AppState,
    model: String,
    generate_request: GenerateRequest,
) -> Result<Sse<KeepAliveStream<Streamer>>, (StatusCode, Json<Value>)> {
    let (sender, receiver) = flume::unbounded();
    if let Err(send_error) =
        app_state
            .llm_service_sender
            .send(ServiceRequest::GenerateStreamingRequest(
                generate_request,
                sender,
            ))
    {
        error!("Failed to send request to LLM Service: {}", send_error);
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Failed to send request to LLM Service",
                    "type": "internal_error",
                }
            })),
        ));
    }

    let streamer = Streamer::new(receiver, model);
    let streaming_interval_in_millis = app_state.streaming_interval_in_millis;
    Ok(Sse::new(streamer).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_millis(streaming_interval_in_millis))
            .text("keep-alive-stream"),
    ))
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use metrics::counter;
    use metrics_exporter_prometheus::PrometheusBuilder;
    use tower::ServiceExt;

    use super::*;

    /// An app state whose LLM service is running, together with the receiver that keeps it so.
    fn test_app_state() -> (AppState, mpsc::UnboundedReceiver<ServiceRequest>) {
        let (llm_service_sender, llm_service_receiver) = mpsc::unbounded_channel();
        let (shutdown_signal_sender, _shutdown_signal_receiver) = mpsc::channel(1);

        (
            AppState {
                request_counter: Arc::new(AtomicU64::new(0)),
                llm_service_sender,
                shutdown_signal_sender,
                streaming_interval_in_millis: 100,
            },
            llm_service_receiver,
        )
    }

    /// A handle over a recorder that is not installed globally, so tests stay independent of each
    /// other and of whatever the process-wide recorder is.
    fn test_prometheus_handle() -> PrometheusHandle {
        PrometheusBuilder::new().build_recorder().handle()
    }

    async fn get(router: Router, path: &str) -> (StatusCode, String) {
        let response = router
            .oneshot(
                Request::builder()
                    .uri(path)
                    .body(Body::empty())
                    .expect("Failed to build request"),
            )
            .await
            .expect("Router failed to answer");

        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("Failed to read response body");

        (
            status,
            String::from_utf8(body.to_vec()).expect("Response body is not valid UTF-8"),
        )
    }

    #[tokio::test]
    async fn test_healthz_reports_a_running_service() {
        let (app_state, _llm_service_receiver) = test_app_state();
        let router = build_router(app_state, test_prometheus_handle());

        let (status, body) = get(router, HEALTHZ_PATH).await;

        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("\"status\":\"ok\""), "got body: {body}");
    }

    #[tokio::test]
    async fn test_healthz_reports_a_stopped_service() {
        let (app_state, llm_service_receiver) = test_app_state();
        drop(llm_service_receiver);
        let router = build_router(app_state, test_prometheus_handle());

        let (status, body) = get(router, HEALTHZ_PATH).await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body.contains("unavailable"), "got body: {body}");
    }

    #[tokio::test]
    async fn test_metrics_renders_what_was_recorded() {
        let (app_state, _llm_service_receiver) = test_app_state();
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            counter!("llm_service_requests_total").increment(3);
        });
        let router = build_router(app_state, handle);

        let (status, body) = get(router, METRICS_PATH).await;

        assert_eq!(status, StatusCode::OK);
        assert!(
            body.contains("llm_service_requests_total 3"),
            "got body: {body}"
        );
    }

    #[tokio::test]
    async fn test_chat_completions_is_mounted_as_a_post_route() {
        let (app_state, _llm_service_receiver) = test_app_state();
        let router = build_router(app_state, test_prometheus_handle());

        let (status, _) = get(router, CHAT_COMPLETIONS_PATH).await;

        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn test_unknown_paths_are_not_found() {
        let (app_state, _llm_service_receiver) = test_app_state();
        let router = build_router(app_state, test_prometheus_handle());

        let (status, _) = get(router, "/not-a-route").await;

        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
