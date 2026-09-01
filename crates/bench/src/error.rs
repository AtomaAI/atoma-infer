//! Errors raised by the benchmark harness.

/// The result type of the harness.
pub type Result<T> = std::result::Result<T, BenchError>;

/// Everything that can go wrong while preparing, running, or reporting a benchmark.
///
/// A failed *request* is not an error — it is a recorded outcome that lands in the results. These
/// variants are the failures that stop a run from producing a number at all.
#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    /// The benchmark configuration is missing something or asks for something impossible.
    #[error("Invalid benchmark configuration: {0}")]
    Config(String),

    /// The path given for the configuration names no file. Reported apart from
    /// [`BenchError::ConfigFile`] because an absent TOML source contributes nothing silently,
    /// which would otherwise surface as every field being missing.
    #[error("No benchmark configuration at {}", .0.display())]
    ConfigMissing(std::path::PathBuf),

    /// The configuration file could not be read or deserialized.
    #[error("Failed to load the benchmark configuration: {0}")]
    ConfigFile(Box<figment::Error>),

    /// A workload could not be built from its source data.
    #[error("Failed to build the workload: {0}")]
    Workload(String),

    /// The tokenizer could not be loaded or used.
    #[error("Tokenizer failure: {0}")]
    Tokenizer(String),

    /// An HTTP request to an engine, or to its metrics endpoint, failed at the transport level.
    #[error("HTTP request to {url} failed: {source}")]
    Http {
        /// The URL that was being requested.
        url: String,
        /// The transport-level failure.
        #[source]
        source: reqwest::Error,
    },

    /// An engine answered, but not in the shape the harness expects.
    #[error("Unexpected response from {url}: {reason}")]
    Protocol {
        /// The URL that answered.
        url: String,
        /// What was wrong with the answer.
        reason: String,
    },

    /// The pinned baseline engine could not be started, or did not become ready.
    #[error("Baseline engine failure: {0}")]
    Baseline(String),

    /// A results artifact could not be read, written, or rendered.
    #[error("Results artifact failure: {0}")]
    Results(String),

    /// Reading or writing a file failed.
    #[error("I/O failure on {path}: {source}")]
    Io {
        /// The path being read or written.
        path: String,
        /// The underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// JSON could not be parsed or serialized.
    #[error("JSON failure: {0}")]
    Json(#[from] serde_json::Error),
}

impl BenchError {
    /// Wraps an I/O failure with the path it happened on.
    pub fn io(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_string(),
            source,
        }
    }

    /// Wraps a transport failure with the URL it happened on.
    pub fn http(url: impl std::fmt::Display, source: reqwest::Error) -> Self {
        Self::Http {
            url: url.to_string(),
            source,
        }
    }

    /// Reports a well-formed response that says the wrong thing.
    pub fn protocol(url: impl std::fmt::Display, reason: impl std::fmt::Display) -> Self {
        Self::Protocol {
            url: url.to_string(),
            reason: reason.to_string(),
        }
    }
}
