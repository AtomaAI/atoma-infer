//! The pinned vLLM baseline.
//!
//! The benchmark protocol pins the competitor by name and version at rung kickoff and runs it in
//! its documented recommended configuration, which for vLLM's OpenAI server is its defaults plus
//! the model. The harness therefore starts vLLM itself, from a pinned image tag, and records both
//! the tag and the version the server reports — a run whose baseline cannot be named is not a
//! comparison.

use std::{path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::error::{BenchError, Result};

/// The vLLM release pinned for rung 0: the latest stable at kickoff (2026-07-27).
pub const PINNED_IMAGE: &str = "vllm/vllm-openai:v0.26.0";

/// Where vLLM's image caches Hugging Face weights.
const CONTAINER_HF_CACHE: &str = "/root/.cache/huggingface";

/// The port vLLM's OpenAI server listens on inside the container.
const CONTAINER_PORT: u16 = 8000;

/// How often readiness is polled while the baseline loads its weights.
const READINESS_POLL: Duration = Duration::from_secs(2);

/// How the pinned baseline is started.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VllmBaseline {
    /// The pinned image, tag included.
    pub image: String,
    /// The model the baseline serves. The same weights the engine under test serves.
    pub model: String,
    /// Host port the container's OpenAI server is published on.
    pub port: u16,
    /// Name given to the container, so a run can stop what it started.
    pub container_name: String,
    /// Value of `docker run --gpus`.
    pub gpus: String,
    /// Value of `docker run --shm-size`, which vLLM needs for its worker IPC.
    pub shm_size: String,
    /// Host Hugging Face cache to mount, so a run does not re-download weights.
    pub hugging_face_cache: Option<PathBuf>,
    /// Arguments appended to `vllm serve`. Empty is the documented recommended configuration.
    pub server_args: Vec<String>,
    /// How long the baseline may take to load its weights and answer `/health`.
    pub startup_timeout_seconds: u64,
}

impl Default for VllmBaseline {
    fn default() -> Self {
        Self {
            image: PINNED_IMAGE.to_string(),
            model: String::new(),
            port: CONTAINER_PORT,
            container_name: "atoma-bench-vllm".to_string(),
            gpus: "all".to_string(),
            shm_size: "16g".to_string(),
            hugging_face_cache: None,
            server_args: Vec::new(),
            startup_timeout_seconds: 900,
        }
    }
}

impl VllmBaseline {
    /// The version this baseline is pinned to, taken from the image tag.
    pub fn pinned_version(&self) -> &str {
        self.image
            .rsplit_once(':')
            .map_or("", |(_, tag)| tag.trim_start_matches('v'))
    }

    /// The base URL the baseline serves on.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Checks the baseline is pinned to a version at all.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Config`] if the image carries no tag, or a moving one.
    pub fn validate(&self) -> Result<()> {
        if self.pinned_version().is_empty() || self.pinned_version() == "latest" {
            return Err(BenchError::Config(format!(
                "The baseline image `{}` is not pinned to a version; the protocol pins the \
                 competitor at rung kickoff",
                self.image
            )));
        }
        if self.model.is_empty() {
            return Err(BenchError::Config(
                "The baseline has no model to serve".to_string(),
            ));
        }
        Ok(())
    }

    /// The command that starts the baseline.
    ///
    /// Everything before the image is docker's; everything after it is `vllm serve`'s.
    pub fn docker_command(&self) -> Vec<String> {
        let mut command = vec![
            "docker".to_string(),
            "run".to_string(),
            "--rm".to_string(),
            "--gpus".to_string(),
            self.gpus.clone(),
            "--shm-size".to_string(),
            self.shm_size.clone(),
            "--name".to_string(),
            self.container_name.clone(),
            "-p".to_string(),
            format!("{}:{CONTAINER_PORT}", self.port),
        ];

        if let Some(cache) = &self.hugging_face_cache {
            command.push("-v".to_string());
            command.push(format!("{}:{CONTAINER_HF_CACHE}", cache.display()));
        }
        // Forwarded by name, never by value: the recorded command has to be the same whatever
        // the operator's environment holds, and a token must not land in a published artifact.
        command.push("-e".to_string());
        command.push("HF_TOKEN".to_string());

        command.push(self.image.clone());
        command.push("--model".to_string());
        command.push(self.model.clone());
        command.push("--port".to_string());
        command.push(CONTAINER_PORT.to_string());
        command.extend(self.server_args.iter().cloned());
        command
    }

    /// What the results artifact records about this baseline.
    pub fn recorded_config(&self) -> serde_json::Value {
        serde_json::json!({
            "image": self.image,
            "model": self.model,
            "server_args": self.server_args,
            "docker_command": self.docker_command().join(" "),
        })
    }

    /// Checks the running server is the pinned version.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Baseline`] if it reports a different one.
    pub fn check_is_pinned_version(&self, reported: &str) -> Result<()> {
        if reported.trim_start_matches('v') == self.pinned_version() {
            return Ok(());
        }

        Err(BenchError::Baseline(format!(
            "The baseline reports version {reported}, but the run is pinned to {} ({}); the \
             results would name a baseline that did not run",
            self.pinned_version(),
            self.image
        )))
    }

    /// Starts the baseline and waits for it to answer, returning the version it reports.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Baseline`] if docker cannot start it, if it does not become ready
    /// within the configured timeout, or if it is not the pinned version.
    pub async fn start(&self) -> Result<RunningBaseline> {
        self.validate()?;
        let command = self.docker_command();
        info!(
            command = command.join(" "),
            "Starting the pinned vLLM baseline"
        );

        let container = tokio::process::Command::new(&command[0])
            .args(&command[1..])
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| {
                BenchError::Baseline(format!("Failed to run `{}`: {error}", command.join(" ")))
            })?;

        let version = match self.await_ready().await {
            Ok(version) => version,
            Err(error) => {
                remove_container(&self.container_name).await;
                return Err(error);
            }
        };

        Ok(RunningBaseline {
            container_name: self.container_name.clone(),
            reported_version: version,
            container,
        })
    }

    /// Polls the baseline until it is serving, then reads the version it reports.
    async fn await_ready(&self) -> Result<String> {
        let http = reqwest::Client::new();
        let health_url = format!("{}/health", self.base_url());
        let deadline =
            std::time::Instant::now() + Duration::from_secs(self.startup_timeout_seconds);

        while std::time::Instant::now() < deadline {
            let healthy = http
                .get(&health_url)
                .send()
                .await
                .is_ok_and(|response| response.status().is_success());
            if healthy {
                let version = self.read_version(&http).await?;
                self.check_is_pinned_version(&version)?;
                info!(version, "The pinned vLLM baseline is serving");
                return Ok(version);
            }
            tokio::time::sleep(READINESS_POLL).await;
        }

        Err(BenchError::Baseline(format!(
            "The baseline did not answer {health_url} within {} s",
            self.startup_timeout_seconds
        )))
    }

    /// Reads the version the running server reports.
    async fn read_version(&self, http: &reqwest::Client) -> Result<String> {
        let url = format!("{}/version", self.base_url());
        let body = http
            .get(&url)
            .send()
            .await
            .map_err(|error| BenchError::http(&url, error))?
            .json::<serde_json::Value>()
            .await
            .map_err(|error| BenchError::http(&url, error))?;

        reported_version(&body, &url)
    }
}

/// Removes a container, whatever state it is in.
async fn remove_container(container_name: &str) {
    let removed = tokio::process::Command::new("docker")
        .args(["rm", "-f", container_name])
        .output()
        .await;

    if let Err(error) = removed {
        warn!(
            container = container_name,
            %error,
            "Failed to remove the baseline container; remove it by hand before the next run"
        );
    }
}

/// Reads the version out of a `/version` response.
fn reported_version(body: &serde_json::Value, url: &str) -> Result<String> {
    body.get("version")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| BenchError::protocol(url, format!("no `version` field in {body}")))
}

/// A baseline that is serving.
pub struct RunningBaseline {
    /// Name of the container serving it.
    container_name: String,
    /// The version the server reported, which the results artifact records.
    reported_version: String,
    /// The `docker run` process, killed if this struct is dropped without being stopped.
    container: tokio::process::Child,
}

impl RunningBaseline {
    /// The version the running baseline reports.
    pub fn reported_version(&self) -> &str {
        &self.reported_version
    }

    /// Stops the baseline.
    pub async fn stop(mut self) {
        remove_container(&self.container_name).await;
        if let Err(error) = self.container.wait().await {
            warn!(%error, "The baseline's docker process did not exit cleanly");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline() -> VllmBaseline {
        VllmBaseline {
            model: "meta-llama/Meta-Llama-3-8B-Instruct".to_string(),
            ..VllmBaseline::default()
        }
    }

    #[test]
    fn test_the_pinned_version_is_the_image_tag() {
        assert_eq!(baseline().pinned_version(), "0.26.0");
        assert_eq!(
            VllmBaseline {
                image: "vllm/vllm-openai:v0.25.1".to_string(),
                ..baseline()
            }
            .pinned_version(),
            "0.25.1"
        );
    }

    /// An untagged image is not a pinned baseline: `latest` would silently re-baseline the
    /// comparison the next time the run happens.
    #[test]
    fn test_an_untagged_image_is_rejected() {
        let error = VllmBaseline {
            image: "vllm/vllm-openai".to_string(),
            ..baseline()
        }
        .validate()
        .expect_err("An untagged image is not pinned");

        assert!(matches!(error, BenchError::Config(_)), "{error}");
    }

    #[test]
    fn test_the_docker_command_serves_the_configured_model_on_the_configured_port() {
        let command = baseline().docker_command();

        assert_eq!(command[0], "docker");
        assert!(command.contains(&"--gpus".to_string()));
        assert!(command.contains(&"vllm/vllm-openai:v0.26.0".to_string()));
        assert!(command.contains(&"meta-llama/Meta-Llama-3-8B-Instruct".to_string()));
        assert!(
            command.windows(2).any(|pair| pair == ["-p", "8000:8000"]),
            "{command:?}"
        );
    }

    /// The baseline runs vLLM's documented defaults unless the protocol's recommended config says
    /// otherwise, and whatever it says has to reach the server's own argument list.
    #[test]
    fn test_recommended_configuration_arguments_are_passed_to_the_server() {
        let command = VllmBaseline {
            server_args: vec!["--max-num-seqs".to_string(), "256".to_string()],
            ..baseline()
        }
        .docker_command();

        let model = command
            .iter()
            .position(|argument| argument == "--model")
            .expect("The model argument is missing");
        assert!(
            command[model..]
                .windows(2)
                .any(|pair| pair == ["--max-num-seqs", "256"]),
            "server arguments must follow the image, not go to docker: {command:?}"
        );
    }

    #[test]
    fn test_a_hugging_face_cache_is_mounted_when_one_is_configured() {
        let command = VllmBaseline {
            hugging_face_cache: Some("/data/hf".into()),
            ..baseline()
        }
        .docker_command();

        assert!(
            command
                .windows(2)
                .any(|pair| pair == ["-v", "/data/hf:/root/.cache/huggingface"]),
            "{command:?}"
        );
    }

    #[test]
    fn test_the_reported_version_is_read_from_the_version_endpoint() {
        let reported = reported_version(&serde_json::json!({ "version": "0.26.0" }), "/version")
            .expect("Failed to read the reported version");

        assert_eq!(reported, "0.26.0");
    }

    #[test]
    fn test_a_version_endpoint_without_a_version_is_an_error() {
        let error = reported_version(&serde_json::json!({ "build": "abc" }), "/version")
            .expect_err("A response without a version is unusable");

        assert!(matches!(error, BenchError::Protocol { .. }), "{error}");
    }

    /// The artifact records the version the server reported; if that is not the pinned one, the
    /// table would name a baseline that never ran.
    #[test]
    fn test_a_server_that_is_not_the_pinned_version_fails_the_run() {
        assert!(baseline().check_is_pinned_version("0.26.0").is_ok());
        assert!(baseline().check_is_pinned_version("v0.26.0").is_ok());

        let error = baseline()
            .check_is_pinned_version("0.25.1")
            .expect_err("A different version must not be reported as the pinned baseline");
        assert!(matches!(error, BenchError::Baseline(_)), "{error}");
    }

    #[test]
    fn test_the_engine_configuration_records_what_the_baseline_ran() {
        let recorded = baseline().recorded_config();

        assert_eq!(recorded["image"], "vllm/vllm-openai:v0.26.0");
        assert_eq!(recorded["model"], "meta-llama/Meta-Llama-3-8B-Instruct");
        assert_eq!(
            recorded["server_args"],
            serde_json::json!([] as [String; 0]),
            "the documented default configuration is recorded as the empty argument list"
        );
    }
}
