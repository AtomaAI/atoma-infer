//! Driving a run.
//!
//! One run: offer the workload's requests at their arrival times, whether or not earlier ones have
//! answered, while the KV-leak probe samples the engine's free-block gauge. What comes out is one
//! [`RunResult`] — the run's numbers and the probe's verdict.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tracing::{info, warn};

use crate::{
    client::OpenAiClient,
    config::{BenchConfig, EngineConfig},
    error::{BenchError, Result},
    kv_probe::{self, KvProbe},
    record::{RequestRecord, RunSummary},
    report::{EngineResults, RunResult},
    vllm::VllmBaseline,
    workload::BenchRequest,
};

/// An engine the harness can drive, however it was started.
#[derive(Clone, Debug)]
pub struct EngineTarget {
    /// The engine's name in the table.
    pub name: String,
    /// The version that ran.
    pub version: String,
    /// Where it serves its OpenAI surface.
    pub base_url: String,
    /// Where it exposes Prometheus metrics, if it does.
    pub metrics_url: Option<String>,
    /// The model id sent with every request.
    pub model: String,
    /// Bearer token, if it wants one.
    pub api_key: Option<String>,
    /// The configuration recorded in the artifact.
    pub config: serde_json::Value,
}

impl EngineTarget {
    /// The engine under test, as its configuration describes it.
    pub fn under_test(engine: &EngineConfig) -> Self {
        Self {
            name: engine.name.clone(),
            version: engine.resolved_version(),
            base_url: engine.base_url.clone(),
            metrics_url: engine.metrics_url.clone(),
            model: engine.model.clone(),
            api_key: engine.api_key.clone(),
            config: engine.recorded_config(),
        }
    }

    /// The pinned baseline, named by the version the running server reported.
    ///
    /// vLLM publishes no free-KV-block gauge, so its runs are unwatched rather than unjudged.
    pub fn baseline(baseline: &VllmBaseline, reported_version: &str) -> Self {
        Self {
            name: format!("vLLM {reported_version}"),
            version: reported_version.to_string(),
            base_url: baseline.base_url(),
            metrics_url: None,
            model: baseline.model.clone(),
            api_key: None,
            config: baseline.recorded_config(),
        }
    }
}

/// Runs a workload against one engine as many times as the protocol asks for.
///
/// A warmup request goes first and is not recorded: the first request an engine sees pays for
/// whatever it does lazily, and the run is meant to measure steady state.
///
/// # Errors
///
/// Returns [`BenchError::Baseline`] if the engine does not answer the warmup request. Failures
/// during a run are recorded, not raised.
pub async fn run_engine(
    target: &EngineTarget,
    config: &BenchConfig,
    requests: &[BenchRequest],
) -> Result<EngineResults> {
    let client = Arc::new(OpenAiClient::new(
        &target.base_url,
        target.model.clone(),
        target.api_key.clone(),
        config.run.request_timeout(),
    )?);

    warm_up(&client, target, requests).await?;

    let mut runs = Vec::with_capacity(config.run.runs);
    for index in 0..config.run.runs {
        info!(engine = target.name, run = index, "Starting a run");
        runs.push(run_once(&client, target, config, requests).await);
        info!(
            engine = target.name,
            run = index,
            goodput = runs[index].summary.goodput_per_second,
            "Run finished"
        );

        if index + 1 < config.run.runs {
            tokio::time::sleep(config.run.cooldown()).await;
        }
    }

    Ok(EngineResults {
        name: target.name.clone(),
        version: target.version.clone(),
        config: target.config.clone(),
        runs,
    })
}

/// Sends one unrecorded request, failing the whole comparison if the engine cannot answer it.
async fn warm_up(
    client: &OpenAiClient,
    target: &EngineTarget,
    requests: &[BenchRequest],
) -> Result<()> {
    let Some(first) = requests.first() else {
        return Err(BenchError::Config(
            "The workload produced no requests".to_string(),
        ));
    };

    let warmup = BenchRequest {
        id: format!("{}-warmup", target.name),
        arrival: Duration::ZERO,
        ..first.clone()
    };
    let record = client.stream_completion(&warmup).await;

    match record.failure_reason() {
        None => Ok(()),
        Some(reason) => Err(BenchError::Baseline(format!(
            "{} did not answer the warmup request at {}: {reason}",
            target.name, target.base_url
        ))),
    }
}

/// Offers the workload once, sampling the KV pool throughout.
async fn run_once(
    client: &Arc<OpenAiClient>,
    target: &EngineTarget,
    config: &BenchConfig,
    requests: &[BenchRequest],
) -> RunResult {
    // The probe takes its baseline sample before any load is offered, so it is started and awaited
    // before `offer` rather than alongside it.
    let probe = match target.metrics_url.as_ref().zip(config.kv_probe.gauge()) {
        Some((metrics_url, gauge)) => Some(
            KvProbe::new(
                metrics_url.clone(),
                gauge.to_string(),
                config.kv_probe.interval(),
            )
            .start()
            .await,
        ),
        None => None,
    };

    let started = Instant::now();
    let records = offer(client, requests).await;
    let duration = started.elapsed();

    let verdict = match probe {
        Some(probe) => Some(kv_probe::verdict(&probe.finish().await, &config.kv_probe)),
        None => None,
    };
    if verdict
        .as_ref()
        .is_some_and(|verdict| !verdict.run_passes())
    {
        warn!(
            engine = target.name,
            ?verdict,
            "The KV-leak probe did not pass this run"
        );
    }

    RunResult {
        summary: RunSummary::summarize(&records, duration, &config.slo),
        kv_probe: verdict,
    }
}

/// Sends every request at its arrival time and collects what each one measured.
///
/// Each request is spawned as its own task and waits for its own arrival, so a slow engine delays
/// answers without delaying the offered load — which is the difference between measuring goodput
/// at a rate and measuring the engine's own pace. Spawning also spreads the harness's own decoding
/// work across the runtime's threads, so the load generator does not become the bottleneck it is
/// measuring.
async fn offer(client: &Arc<OpenAiClient>, requests: &[BenchRequest]) -> Vec<RequestRecord> {
    let started = Instant::now();
    let offered = requests
        .iter()
        .map(|request| {
            let request = request.clone();
            let client = Arc::clone(client);
            let deadline = started + request.arrival;

            tokio::spawn(async move {
                tokio::time::sleep_until(deadline.into()).await;
                client.stream_completion(&request).await
            })
        })
        .collect::<Vec<_>>();

    let mut records = Vec::with_capacity(requests.len());
    for (request, offered) in requests.iter().zip(offered) {
        records.push(offered.await.unwrap_or_else(|error| {
            RequestRecord::failed(
                request.id.clone(),
                request.prompt_tokens,
                format!("the harness task for this request did not finish: {error}"),
                Duration::ZERO,
            )
        }));
    }

    records
}
