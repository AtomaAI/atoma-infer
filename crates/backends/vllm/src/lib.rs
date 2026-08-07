//! This crate contains logic for fast inference servince, based on PagedAttention and the vLLM
//! implementation. We refer the reader to https://arxiv.org/pdf/2309.06180 for the detailed architecture of the service. We were highly inspired by the complete, in production, Python implementation
//! of vLLM, in https://github.com/vllm-project/vllm.

mod admission;
pub mod block;
pub mod block_allocator;
pub mod block_manager;
pub mod config;
pub mod egress;
pub mod error;
pub mod llm_engine;
pub mod llm_service;
#[cfg(feature = "cuda")]
pub mod model_executor;
#[cfg(feature = "cuda")]
pub mod models;
pub mod output;
pub mod policy;
pub mod request;
pub mod sampling;
pub mod scheduler;
pub mod sequence;
#[cfg(test)]
mod tests;
pub mod tokenizer;
pub mod types;
pub mod validation;
#[cfg(feature = "cuda")]
pub mod worker;

pub use error::LlmServiceError;
#[cfg(feature = "cuda")]
pub use llm_service::LlmService;
pub use output::{GenerateRequestOutput, GenerateStreamingOutput, StreamResponse};
pub use request::ServiceRequest;
pub use types::{GenerateParameters, GenerateRequest};
pub use validation::Validation;

#[cfg(all(feature = "cuda", not(feature = "nccl")))]
pub use models::llama::LlamaModel;
#[cfg(all(feature = "cuda", feature = "nccl"))]
pub use models::llama_nccl::LlamaModel;
