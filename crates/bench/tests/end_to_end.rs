//! The harness driven end to end against a stub engine.
//!
//! The stub speaks the same OpenAI streaming surface and publishes the same free-block gauge as
//! the engine under test, so these tests cover what the unit tests cannot: arrivals actually
//! firing over the wire, tokens actually being timed, and the KV-leak probe actually failing a run
//! whose pool drains.

use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use atoma_bench::{
    config::{BenchConfig, EngineConfig, RunConfig},
    goodput::Slo,
    kv_probe::{KvProbe, KvProbeConfig, KvProbeVerdict},
    report::EngineResults,
    runner::{run_engine, EngineTarget},
    vllm::VllmBaseline,
    workload::{LoadPlan, LongContextSpec, WordVocabulary, WorkloadSpec},
};
use axum::{
    body::{Body, Bytes},
    extract::State,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};

/// Tokens the stub streams per answer.
const TOKENS_PER_ANSWER: usize = 6;

/// The stub's KV pool, in blocks.
const POOL_BLOCKS: u64 = 512;

/// Blocks the stub holds per request while it is answering.
const BLOCKS_PER_REQUEST: u64 = 4;

/// How the stub engine behaves.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Behaviour {
    /// Answers every request and returns every block.
    Healthy,
    /// Answers every request and keeps some of the blocks of each one — the rung-0 P0.
    Leaking,
    /// Answers the first request, then refuses everything.
    RefusesAfterWarmup,
}

/// The stub engine's state.
struct StubState {
    /// How it behaves.
    behaviour: Behaviour,
    /// Requests currently being answered.
    in_flight: AtomicU64,
    /// Requests answered so far.
    answered: AtomicU64,
    /// Times `/metrics` has been scraped, so a test can tell when the probe read the gauge.
    scrapes: AtomicU64,
}

impl StubState {
    /// The free-block count the stub publishes: blocks are held while a request is in flight, and
    /// a leaking engine also keeps one per request it has finished.
    fn free_blocks(&self) -> u64 {
        let held = self.in_flight.load(Ordering::Relaxed) * BLOCKS_PER_REQUEST;
        let leaked = match self.behaviour {
            Behaviour::Leaking => self.answered.load(Ordering::Relaxed),
            Behaviour::Healthy | Behaviour::RefusesAfterWarmup => 0,
        };

        POOL_BLOCKS.saturating_sub(held + leaked)
    }
}

/// How long the stub takes to produce its first token.
const PREFILL: Duration = Duration::from_millis(20);

/// How long the stub takes per token after the first.
const DECODE_STEP: Duration = Duration::from_millis(5);

/// Streams an answer token by token, so the harness times a real stream rather than one response
/// that happens to contain several events.
async fn completions(State(state): State<Arc<StubState>>) -> Response {
    if state.behaviour == Behaviour::RefusesAfterWarmup
        && state.answered.load(Ordering::Relaxed) > 0
    {
        state.answered.fetch_add(1, Ordering::Relaxed);
        return (axum::http::StatusCode::SERVICE_UNAVAILABLE, "overloaded").into_response();
    }

    state.in_flight.fetch_add(1, Ordering::Relaxed);
    let stream = futures::stream::unfold(0, move |index| {
        let state = Arc::clone(&state);
        async move {
            match index {
                0 => tokio::time::sleep(PREFILL).await,
                index if index < TOKENS_PER_ANSWER => tokio::time::sleep(DECODE_STEP).await,
                index if index > TOKENS_PER_ANSWER => return None,
                _ => {
                    state.in_flight.fetch_sub(1, Ordering::Relaxed);
                    state.answered.fetch_add(1, Ordering::Relaxed);
                    return Some((
                        Ok::<_, std::io::Error>(Bytes::from_static(b"data: [DONE]\n\n")),
                        index + 1,
                    ));
                }
            }

            let content = format!("t{index} ");
            let token = serde_json::json!({ "choices": [{ "delta": { "content": content } }] });
            Some((Ok(Bytes::from(format!("data: {token}\n\n"))), index + 1))
        }
    });

    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        Body::from_stream(stream),
    )
        .into_response()
}

/// The gauge the stub engine publishes its free-block count on, standing in for the name a real
/// engine publishes and an operator puts in `kv_probe.metric`.
const STUB_FREE_BLOCKS_METRIC: &str = "atoma_kv_free_gpu_blocks";

/// Publishes the free-block gauge in the Prometheus text format.
async fn metrics(State(state): State<Arc<StubState>>) -> String {
    state.scrapes.fetch_add(1, Ordering::Relaxed);
    format!(
        "# TYPE {metric} gauge\n{metric} {value}\n",
        metric = STUB_FREE_BLOCKS_METRIC,
        value = state.free_blocks()
    )
}

/// Starts a stub engine and returns its state alongside the address it serves on.
async fn start_stub_with_state(behaviour: Behaviour) -> (SocketAddr, Arc<StubState>) {
    let state = Arc::new(StubState {
        behaviour,
        in_flight: AtomicU64::new(0),
        answered: AtomicU64::new(0),
        scrapes: AtomicU64::new(0),
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(completions))
        .route("/metrics", get(metrics))
        .with_state(Arc::clone(&state));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind the stub engine");
    let address = listener
        .local_addr()
        .expect("The stub engine has no address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("The stub engine stopped");
    });

    (address, state)
}

/// Starts a stub engine and returns the address it serves on.
async fn start_stub(behaviour: Behaviour) -> SocketAddr {
    start_stub_with_state(behaviour).await.0
}

/// A configuration that offers 24 requests at 60/s and samples the pool every 10 ms.
fn config(address: SocketAddr, runs: usize) -> BenchConfig {
    BenchConfig {
        run: RunConfig {
            runs,
            hardware: "stub".to_string(),
            tokenizer: "stub".to_string(),
            request_timeout_seconds: 10,
            cooldown_seconds: 0,
        },
        load: LoadPlan {
            num_requests: 24,
            request_rate_per_second: 60.0,
            seed: 5,
        },
        slo: Slo {
            ttft_ms: 2_000.0,
            itl_ms: 50.0,
        },
        workload: WorkloadSpec::LongContext(LongContextSpec {
            input_tokens: 16,
            output_tokens: TOKENS_PER_ANSWER as u32,
            range_ratio: 0.0,
        }),
        engine: EngineConfig {
            name: "stub-engine".to_string(),
            version: Some("test".to_string()),
            base_url: format!("http://{address}"),
            metrics_url: Some(format!("http://{address}/metrics")),
            model: "stub-model".to_string(),
            config_path: None,
            api_key: None,
        },
        baseline: VllmBaseline {
            model: "stub-model".to_string(),
            ..VllmBaseline::default()
        },
        kv_probe: KvProbeConfig {
            metric: Some(STUB_FREE_BLOCKS_METRIC.to_string()),
            interval_ms: 10,
            ..KvProbeConfig::default()
        },
    }
}

/// What the probe concluded about one run of a watched engine.
fn probe_verdict(engine: &EngineResults, run: usize) -> &KvProbeVerdict {
    engine.runs[run]
        .kv_probe
        .as_ref()
        .expect("The stub engine publishes a gauge, so its runs are watched")
}

/// Drives a stub engine through the harness.
async fn drive(behaviour: Behaviour, runs: usize) -> EngineResults {
    let address = start_stub(behaviour).await;
    let config = config(address, runs);
    let requests = config
        .workload
        .build(&WordVocabulary, &config.load)
        .expect("Failed to build the workload");

    run_engine(
        &EngineTarget::under_test(&config.engine),
        &config,
        &requests,
    )
    .await
    .expect("The stub engine did not complete its runs")
}

#[tokio::test(flavor = "multi_thread")]
async fn test_a_run_measures_every_request_it_offers() {
    let engine = drive(Behaviour::Healthy, 1).await;

    let summary = &engine.runs[0].summary;
    assert_eq!(summary.num_requests, 24);
    assert_eq!(summary.completed, 24, "the stub answered every request");
    assert_eq!(summary.failed, 0);
    assert_eq!(summary.output_tokens, 24 * TOKENS_PER_ANSWER);
    assert!(summary.ttft_ms.p50_ms > 0.0, "{summary:?}");
    assert_eq!(
        summary.itl_ms.count as usize,
        24 * (TOKENS_PER_ANSWER - 1),
        "every gap between streamed tokens must be recorded"
    );
    assert!(
        (summary.itl_ms.p50_ms - DECODE_STEP.as_secs_f64() * 1_000.0).abs() < 5.0,
        "inter-token latency must reflect the stub's decode step: {summary:?}"
    );
    assert!(
        summary.ttft_ms.p50_ms >= PREFILL.as_secs_f64() * 1_000.0,
        "time to first token must cover the stub's prefill: {summary:?}"
    );
    assert!(
        summary.goodput_per_second > 0.0,
        "a stub that answers inside the SLO must have goodput: {summary:?}"
    );
}

/// The offered rate is what the harness controls; a run that took far longer than the arrival
/// schedule would mean load was being held back by the client rather than the engine.
#[tokio::test(flavor = "multi_thread")]
async fn test_load_is_offered_open_loop_at_the_configured_rate() {
    let engine = drive(Behaviour::Healthy, 1).await;

    let duration = engine.runs[0].summary.duration_seconds;
    assert!(
        (0.3..1.5).contains(&duration),
        "24 requests at 60/s should take about 0.4 s, took {duration} s"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_the_kv_probe_passes_a_run_against_an_engine_that_returns_its_blocks() {
    let engine = drive(Behaviour::Healthy, 1).await;

    assert!(
        probe_verdict(&engine, 0).run_passes(),
        "{:?}",
        engine.runs[0].kv_probe
    );
}

/// The acceptance criterion of the probe: an engine whose pool drains fails the run, even though
/// every request it answered looked fine.
#[tokio::test(flavor = "multi_thread")]
async fn test_the_kv_probe_fails_a_run_against_a_leaking_engine() {
    let engine = drive(Behaviour::Leaking, 1).await;

    assert_eq!(
        engine.runs[0].summary.completed, 24,
        "the leak is invisible in the latencies"
    );
    assert!(
        matches!(probe_verdict(&engine, 0), KvProbeVerdict::Leak { .. }),
        "{:?}",
        engine.runs[0].kv_probe
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_refused_requests_are_recorded_as_failures() {
    let engine = drive(Behaviour::RefusesAfterWarmup, 1).await;

    let summary = &engine.runs[0].summary;
    assert_eq!(summary.completed, 0);
    assert_eq!(summary.failed, 24);
    assert_eq!(
        summary.goodput_per_second, 0.0,
        "refused requests must not count towards goodput"
    );
    assert_eq!(summary.ttft_ms.count, 0, "a refusal has no first token");
}

/// The baseline sample is what the drain check judges a run against, so it has to be read before
/// the caller offers any load. Leaving it to the probe's spawned task would race the run's first
/// arrivals and record a pool that had already handed blocks out.
#[tokio::test(flavor = "multi_thread")]
async fn test_the_probe_reads_its_baseline_before_it_starts_sampling() {
    let (address, state) = start_stub_with_state(Behaviour::Healthy).await;

    let probe = KvProbe::new(
        format!("http://{address}/metrics"),
        STUB_FREE_BLOCKS_METRIC.to_string(),
        Duration::from_secs(60),
    )
    .start()
    .await;

    assert_eq!(
        state.scrapes.load(Ordering::Relaxed),
        1,
        "the gauge must have been read by the time `start` returns"
    );

    let samples = probe.finish().await;
    assert_eq!(
        samples.first().map(|sample| sample.free_blocks),
        Some(POOL_BLOCKS),
        "the baseline is the whole pool, measured before any load: {samples:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_the_protocol_run_count_is_honoured() {
    let engine = drive(Behaviour::Healthy, 3).await;

    assert_eq!(engine.runs.len(), 3);
    assert!(engine.runs.iter().all(|run| run.summary.completed == 24
        && run
            .kv_probe
            .as_ref()
            .is_some_and(atoma_bench::kv_probe::KvProbeVerdict::run_passes)));
}
