//! The benchmark's configuration.
//!
//! One TOML file describes a comparison end to end: the workload, the load it is offered at, the
//! service level goodput is measured against, how to reach the engine under test, and which pinned
//! baseline to start. Both engines' configurations are copied into the results artifact, so a
//! published table always carries the setup that produced it.

use std::{path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

use crate::{
    error::{BenchError, Result},
    goodput::Slo,
    kv_probe::KvProbeConfig,
    vllm::VllmBaseline,
    workload::{LoadPlan, WorkloadSpec},
};

/// A whole comparison.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchConfig {
    /// How the comparison is run and where its artifacts land.
    pub run: RunConfig,
    /// The load offered to both engines.
    pub load: LoadPlan,
    /// The service level goodput is measured against.
    pub slo: Slo,
    /// The workload both engines are offered.
    pub workload: WorkloadSpec,
    /// The engine under test.
    pub engine: EngineConfig,
    /// The pinned baseline.
    pub baseline: VllmBaseline,
    /// How the KV pool is watched during a run.
    #[serde(default)]
    pub kv_probe: KvProbeConfig,
}

impl BenchConfig {
    /// Reads a configuration from a TOML file.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::ConfigFile`] if the file cannot be read or does not describe a
    /// benchmark, and [`BenchError::Config`] if what it describes cannot be run.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let config: Self = config::Config::builder()
            .add_source(config::File::from(path))
            .build()?
            .try_deserialize()?;

        config.validate()?;
        Ok(config)
    }

    /// Checks the configuration describes a runnable comparison.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Config`] if the protocol's run count is not met, or if the baseline
    /// is not pinned.
    pub fn validate(&self) -> Result<()> {
        if self.run.runs < 3 {
            return Err(BenchError::Config(format!(
                "The protocol reports the median of at least 3 runs, but `run.runs` is {}",
                self.run.runs
            )));
        }
        if self.load.num_requests == 0 {
            return Err(BenchError::Config(
                "A run with no requests measures nothing".to_string(),
            ));
        }
        self.baseline.validate()
    }
}

/// How the comparison is run.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunConfig {
    /// Runs per engine. The protocol reports the median of at least three.
    pub runs: usize,
    /// The hardware both engines run on, recorded in the artifact as the operator declares it.
    pub hardware: String,
    /// The tokenizer prompts are measured with: a `tokenizer.json` path or a Hugging Face model
    /// id.
    pub tokenizer: String,
    /// How long a single request may take before it is recorded as failed.
    #[serde(default = "default_request_timeout_seconds")]
    pub request_timeout_seconds: u64,
    /// How long the harness waits between runs for the engine to settle.
    #[serde(default = "default_cooldown_seconds")]
    pub cooldown_seconds: u64,
}

impl RunConfig {
    /// The per-request timeout.
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_seconds)
    }

    /// The pause between runs.
    pub fn cooldown(&self) -> Duration {
        Duration::from_secs(self.cooldown_seconds)
    }
}

/// A request may take ten minutes before the harness gives up on it: a long-context prefill behind
/// a full queue is slow, not hung.
fn default_request_timeout_seconds() -> u64 {
    600
}

/// Long enough for a run's requests to drain and the KV pool to settle before the next one starts.
fn default_cooldown_seconds() -> u64 {
    30
}

/// How to reach the engine under test, and what to record about it.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EngineConfig {
    /// The engine's name in the table.
    pub name: String,
    /// The version that ran. Absent, the current git commit is used.
    #[serde(default)]
    pub version: Option<String>,
    /// Where the engine serves its OpenAI surface.
    pub base_url: String,
    /// Where it exposes Prometheus metrics, which the KV-leak probe samples.
    pub metrics_url: Option<String>,
    /// The model id sent with every request.
    pub model: String,
    /// The engine's own configuration file, copied into the results artifact.
    pub config_path: Option<PathBuf>,
    /// Bearer token, if the engine wants one.
    pub api_key: Option<String>,
}

impl EngineConfig {
    /// The version recorded in the artifact: the configured one, or the current git commit.
    pub fn resolved_version(&self) -> String {
        if let Some(version) = &self.version {
            return version.clone();
        }

        std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|git| git.status.success())
            .map(|git| String::from_utf8_lossy(&git.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// What the results artifact records about this engine: its endpoint, its model, and the
    /// contents of the configuration file it was started with.
    pub fn recorded_config(&self) -> serde_json::Value {
        let engine_config =
            self.config_path
                .as_ref()
                .map(|path| match std::fs::read_to_string(path) {
                    Ok(contents) => contents,
                    Err(error) => format!("<unreadable {}: {error}>", path.display()),
                });

        serde_json::json!({
            "base_url": self.base_url,
            "model": self.model,
            "config_path": self.config_path,
            "config": engine_config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_toml() -> &'static str {
        r#"
[run]
runs = 3
hardware = "1xH100 SXM 80GB"
tokenizer = "meta-llama/Meta-Llama-3-8B-Instruct"

[load]
num_requests = 200
request_rate_per_second = 8.0
seed = 1

[slo]
ttft_ms = 2000.0
itl_ms = 50.0

[workload]
kind = "long-context"
input_tokens = 8000
output_tokens = 256
range_ratio = 0.1

[engine]
name = "atoma-infer"
base_url = "http://127.0.0.1:8080"
metrics_url = "http://127.0.0.1:8080/metrics"
model = "meta-llama/Meta-Llama-3-8B-Instruct"

[baseline]
image = "vllm/vllm-openai:v0.26.0"
model = "meta-llama/Meta-Llama-3-8B-Instruct"
port = 8000
container_name = "atoma-bench-vllm"
gpus = "all"
shm_size = "16g"
server_args = []
startup_timeout_seconds = 900
"#
    }

    fn write_config(directory: &std::path::Path, toml: &str) -> PathBuf {
        let path = directory.join("bench.toml");
        std::fs::write(&path, toml).expect("Failed to write the configuration");
        path
    }

    /// A directory of this process's own, so concurrent test runs on one host do not share files.
    fn temp_dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("atoma-bench-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("Failed to create the temporary directory");
        path
    }

    #[test]
    fn test_a_configuration_file_describes_a_whole_comparison() {
        let path = write_config(&temp_dir("config-whole"), config_toml());

        let config = BenchConfig::load(&path).expect("Failed to load the configuration");

        assert_eq!(config.run.runs, 3);
        assert_eq!(config.load.num_requests, 200);
        assert_eq!(config.workload.name(), "long-context");
        assert_eq!(config.engine.name, "atoma-infer");
        assert_eq!(config.baseline.pinned_version(), "0.26.0");
        assert_eq!(config.kv_probe.metric, "atoma_kv_free_gpu_blocks");
    }

    /// The example is what an operator copies; if it stops matching this struct, the first thing
    /// they see on the rig is a deserialization error.
    #[test]
    fn test_the_example_configuration_loads() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench.example.toml");

        let config = BenchConfig::load(&path).expect("Failed to load bench.example.toml");

        assert_eq!(config.workload.name(), "sharegpt");
        assert_eq!(config.baseline.pinned_version(), "0.26.0");
        assert_eq!(config.kv_probe.metric, "atoma_kv_free_gpu_blocks");
    }

    /// Two runs cannot have a median, and the protocol asks for at least three.
    #[test]
    fn test_a_configuration_with_too_few_runs_is_rejected() {
        let path = write_config(
            &temp_dir("config-runs"),
            &config_toml().replace("runs = 3", "runs = 2"),
        );

        let error = BenchConfig::load(&path).expect_err("Two runs is not a median");

        assert!(matches!(error, BenchError::Config(_)), "{error}");
    }

    #[test]
    fn test_a_configuration_whose_baseline_is_not_pinned_is_rejected() {
        let path = write_config(
            &temp_dir("config-baseline"),
            &config_toml().replace("vllm/vllm-openai:v0.26.0", "vllm/vllm-openai:latest"),
        );

        let error = BenchConfig::load(&path).expect_err("`latest` is not a pinned baseline");

        assert!(matches!(error, BenchError::Config(_)), "{error}");
    }

    #[test]
    fn test_the_engine_configuration_file_is_copied_into_the_artifact() {
        let directory = temp_dir("config-recorded");
        let engine_config = directory.join("engine.toml");
        std::fs::write(&engine_config, "block_size = 16\n").expect("Failed to write");

        let recorded = EngineConfig {
            name: "atoma-infer".to_string(),
            version: Some("abc1234".to_string()),
            base_url: "http://127.0.0.1:8080".to_string(),
            metrics_url: None,
            model: "meta-llama/Meta-Llama-3-8B-Instruct".to_string(),
            config_path: Some(engine_config),
            api_key: None,
        }
        .recorded_config();

        assert_eq!(recorded["config"], "block_size = 16\n");
        assert_eq!(recorded["model"], "meta-llama/Meta-Llama-3-8B-Instruct");
    }

    /// An artifact that names no version is not evidence; the git commit stands in when the
    /// operator does not name one.
    #[test]
    fn test_an_unset_version_falls_back_to_the_git_commit() {
        let engine = EngineConfig {
            name: "atoma-infer".to_string(),
            version: None,
            base_url: "http://127.0.0.1:8080".to_string(),
            metrics_url: None,
            model: "model".to_string(),
            config_path: None,
            api_key: None,
        };

        assert!(!engine.resolved_version().is_empty());
    }
}
