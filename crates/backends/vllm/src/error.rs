use candle_core::{DTypeParseError, Error as CandleError};
use thiserror::Error;
use tokio::sync::mpsc::error::SendError;

use crate::{
    config::{CacheConfigError, SchedulerConfigError},
    output::GenerateRequestOutput,
    request::EngineRequest,
    scheduler::SchedulerError,
    sequence::SequenceError,
    tokenizer::TokenizerError,
    validation::ValidationError,
};

#[cfg(feature = "cuda")]
use crate::model_executor::{ConfigError, ModelLoaderError, ModelThreadError};

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Scheduler error: `{0}`")]
    SchedulerError(#[from] SchedulerError),
    #[error("Sequence error: `{0}`")]
    SequenceError(#[from] SequenceError),
    #[error("Missing sequence output token, id = `{0}`")]
    MissingSequenceOutputToken(u64),
    #[error("Tokenizer error: `{0}`")]
    TokenizerError(String),
    #[error("Send error: `{0}`")]
    SendError(#[from] SendError<Vec<GenerateRequestOutput>>),
    #[error("Model worker died, every pending request was failed: `{0}`")]
    ModelWorkerDead(String),
}

#[derive(Debug, Error)]
pub enum LlmServiceError {
    #[error("Boxed error: `{0}`")]
    BoxedError(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("Broadcast sender error: `{0}`")]
    BroadcastSenderError(String),
    #[error("Cache config error: `{0}`")]
    CacheConfigError(#[from] CacheConfigError),
    #[error("Candle error: `{0}`")]
    CandleError(#[from] CandleError),
    #[cfg(feature = "cuda")]
    #[error("Config error: `{0}`")]
    ConfigError(#[from] ConfigError),
    #[error("DType parse error: `{0}`")]
    DTypeParseError(#[from] DTypeParseError),
    #[cfg(feature = "cuda")]
    #[error("Model loader error: `{0}`")]
    ModelLoaderError(#[from] ModelLoaderError),
    #[cfg(feature = "cuda")]
    #[error("Model thread error: `{0}`")]
    ModelThreadError(#[from] ModelThreadError),
    #[error("Engine error: `{0}`")]
    EngineError(#[from] EngineError),
    #[error("Validation error: `{0}`")]
    ValidationError(#[from] ValidationError),
    #[error("Scheduler error: `{0}`")]
    SchedulerError(#[from] SchedulerError),
    #[error("Scheduler config error: `{0}`")]
    SchedulerConfigError(#[from] SchedulerConfigError),
    #[error("Sequence error: `{0}`")]
    SequenceError(#[from] SequenceError),
    #[error("Send error: `{0}`")]
    SendError(#[from] Box<SendError<EngineRequest>>),
    #[error("Tokenizer error: `{0}`")]
    TokenizerError(#[from] TokenizerError),
}
