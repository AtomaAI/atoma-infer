//! This crate contains logic for fast inference servince, based on PagedAttention and the vLLM
//! implementation. We refer the reader to https://arxiv.org/pdf/2309.06180 for the detailed architecture of the service. We were highly inspired by the complete, in production, Python implementation
//! of vLLM, in https://github.com/vllm-project/vllm.

pub mod block;
pub mod block_allocator;
pub mod block_manager;
pub mod config;
pub mod llm_engine;
pub mod llm_service;
#[cfg(feature = "cuda")]
pub mod model_executor;
#[cfg(feature = "cuda")]
pub mod models;
pub mod policy;
pub mod scheduler;
pub mod sequence;
#[cfg(all(test, feature = "cuda"))]
pub mod tests;
pub mod tokenizer;
pub mod types;
pub mod validation;
#[cfg(feature = "cuda")]
pub mod worker;

pub use llm_engine::{GenerateRequestOutput, GenerateStreamingOutput, StreamResponse};
#[cfg(feature = "cuda")]
pub use llm_service::LlmService;
pub use llm_service::{LlmServiceError, ServiceRequest};
pub use types::{GenerateParameters, GenerateRequest};
pub use validation::Validation;

#[cfg(all(feature = "cuda", not(feature = "nccl")))]
pub use models::llama::LlamaModel;
#[cfg(all(feature = "cuda", feature = "nccl"))]
pub use models::llama_nccl::LlamaModel;
