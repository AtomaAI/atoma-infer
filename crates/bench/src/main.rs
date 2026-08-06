//! The benchmark harness's command line.
//!
//! A comparison is three commands, not one, because both engines want the same GPU:
//!
//! ```shell
//! # 1. with atoma-infer serving:
//! atoma-bench run --config bench.toml --results results/atoma-sharegpt.json
//! # 2. stop atoma-infer, freeing the GPU, then:
//! atoma-bench baseline --config bench.toml --results results/vllm-sharegpt.json
//! # 3. any time after:
//! atoma-bench compare --config bench.toml \
//!     --engine results/atoma-sharegpt.json \
//!     --baseline results/vllm-sharegpt.json \
//!     --results docs/benchmarks/rung0-baseline-sharegpt.json \
//!     --table docs/benchmarks/rung0-baseline-table.md
//! ```
//!
//! `run` alone is also how the KV-leak probe is used as a standing regression guard.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use atoma_bench::{
    config::BenchConfig,
    report::{BenchmarkResults, EngineResults},
    runner::{run_engine, EngineTarget},
    workload::{BenchRequest, HfVocabulary},
};
use clap::{Parser, Subcommand};
use tracing::info;

/// The benchmark harness for `atoma-infer`.
#[derive(Debug, Parser)]
#[command(name = "atoma-bench", version, about)]
struct Cli {
    /// What to do.
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Drives the engine under test, which must already be serving.
    Run {
        /// The benchmark configuration.
        #[arg(long, default_value = "bench.toml")]
        config: PathBuf,
        /// Where to write the engine's runs, as JSON.
        #[arg(long)]
        results: PathBuf,
    },
    /// Starts the pinned vLLM baseline and drives it over the same workload.
    ///
    /// Stop the engine under test first: both engines claim the whole GPU.
    Baseline {
        /// The benchmark configuration.
        #[arg(long, default_value = "bench.toml")]
        config: PathBuf,
        /// Where to write the baseline's runs, as JSON.
        #[arg(long)]
        results: PathBuf,
    },
    /// Combines two run artifacts into the results artifact and renders the table.
    Compare {
        /// The benchmark configuration both engines were driven with.
        #[arg(long, default_value = "bench.toml")]
        config: PathBuf,
        /// The engine under test's runs, from `run`.
        #[arg(long)]
        engine: PathBuf,
        /// The pinned baseline's runs, from `baseline`.
        #[arg(long)]
        baseline: PathBuf,
        /// Where to write the combined results artifact, as JSON.
        #[arg(long)]
        results: PathBuf,
        /// Where to write the rendered table, as markdown.
        #[arg(long)]
        table: PathBuf,
    },
    /// Renders the table from a results artifact that already exists.
    Render {
        /// The results artifact to render.
        #[arg(long)]
        results: PathBuf,
        /// Where to write the rendered table, as markdown.
        #[arg(long)]
        table: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    match Cli::parse().command {
        Command::Run { config, results } => run(&config, &results).await,
        Command::Baseline { config, results } => baseline(&config, &results).await,
        Command::Compare {
            config,
            engine,
            baseline,
            results,
            table,
        } => compare(&config, &engine, &baseline, &results, &table),
        Command::Render { results, table } => render(&results, &table),
    }
}

/// Drives the engine under test and writes what its runs measured.
async fn run(config_path: &Path, results_path: &Path) -> Result<()> {
    let (config, requests) = prepare(config_path)?;

    let engine = run_engine(
        &EngineTarget::under_test(&config.engine),
        &config,
        &requests,
    )
    .await
    .context("The engine under test did not complete its runs")?;

    report_probe_verdicts(&engine);
    write_json(results_path, &engine)
}

/// Starts the pinned baseline, drives it, and stops it again.
async fn baseline(config_path: &Path, results_path: &Path) -> Result<()> {
    let (config, requests) = prepare(config_path)?;
    let baseline = config
        .baseline
        .start()
        .await
        .context("Failed to start the pinned vLLM baseline")?;

    let target = EngineTarget {
        name: format!("vLLM {}", baseline.reported_version()),
        version: baseline.reported_version().to_string(),
        base_url: config.baseline.base_url(),
        // vLLM publishes no free-KV-block gauge, so its runs are unwatched rather than unjudged.
        metrics_url: None,
        model: config.baseline.model.clone(),
        api_key: None,
        config: config.baseline.recorded_config(),
    };

    let runs = run_engine(&target, &config, &requests).await;
    baseline.stop().await;

    write_json(
        results_path,
        &runs.context("The pinned baseline did not complete its runs")?,
    )
}

/// Combines both engines' runs, checks them against the protocol, and renders the table.
///
/// The table is written only if the numbers may be published: a run that leaked KV blocks, or a
/// comparison with too few runs, leaves the artifact for inspection and no table.
fn compare(
    config_path: &Path,
    engine_path: &Path,
    baseline_path: &Path,
    results_path: &Path,
    table_path: &Path,
) -> Result<()> {
    let config = load_config(config_path)?;
    let results = BenchmarkResults {
        generated_at: chrono::Utc::now().to_rfc3339(),
        hardware: config.run.hardware.clone(),
        workload: config.workload.name().to_string(),
        plan: config.load,
        slo: config.slo,
        subject: read_json(engine_path)?,
        baseline: read_json(baseline_path)?,
    };

    write_json(results_path, &results)?;
    results
        .validate()
        .context("The runs do not meet the benchmark protocol, so no table was written")?;

    write_text(table_path, &results.render_document(results_path))
}

/// Re-renders a table from an artifact that already exists.
fn render(results_path: &Path, table_path: &Path) -> Result<()> {
    let results: BenchmarkResults = read_json(results_path)?;

    results
        .validate()
        .context("The artifact does not meet the benchmark protocol, so no table was written")?;
    write_text(table_path, &results.render_document(results_path))
}

/// Loads the configuration and builds the request stream both engines are offered.
fn prepare(config_path: &Path) -> Result<(BenchConfig, Vec<BenchRequest>)> {
    let config = load_config(config_path)?;
    let vocabulary = HfVocabulary::load(&config.run.tokenizer)
        .with_context(|| format!("Failed to load the tokenizer `{}`", config.run.tokenizer))?;
    let requests = config
        .workload
        .build(&vocabulary, &config.load)
        .context("Failed to build the workload")?;

    info!(
        workload = config.workload.name(),
        requests = requests.len(),
        rate = config.load.request_rate_per_second,
        "Built the workload"
    );
    Ok((config, requests))
}

/// Reads the benchmark configuration.
fn load_config(path: &Path) -> Result<BenchConfig> {
    BenchConfig::load(path).with_context(|| format!("Failed to load {}", path.display()))
}

/// Logs what the KV-leak probe concluded about each run.
fn report_probe_verdicts(engine: &EngineResults) {
    for (index, run) in engine.runs.iter().enumerate() {
        info!(
            engine = engine.name,
            run = index,
            verdict = ?run.kv_probe,
            "KV-leak probe"
        );
    }
}

/// Reads a JSON artifact.
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let artifact = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;

    serde_json::from_str(&artifact).with_context(|| {
        format!(
            "{} is not the artifact it was expected to be",
            path.display()
        )
    })
}

/// Writes a JSON artifact, creating the directory it lives in.
fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    write_text(path, &serde_json::to_string_pretty(value)?)
}

/// Writes a file, creating the directory it lives in.
fn write_text(path: &Path, contents: &str) -> Result<()> {
    if let Some(directory) = path.parent() {
        std::fs::create_dir_all(directory)
            .with_context(|| format!("Failed to create {}", directory.display()))?;
    }
    std::fs::write(path, contents)
        .with_context(|| format!("Failed to write {}", path.display()))?;

    info!(path = %path.display(), "Wrote");
    Ok(())
}
