use tokio::sync::oneshot;

use crate::{
    output::{GenerateRequestOutput, StreamResponse},
    sequence::SequenceGroup,
    types::GenerateRequest,
};

/// Requests accepted by the LLM service.
pub enum ServiceRequest {
    /// Generate a complete response.
    GenerateRequest(GenerateRequest, oneshot::Sender<GenerateRequestOutput>),
    /// Generate a streaming response.
    GenerateStreamingRequest(GenerateRequest, flume::Sender<StreamResponse>),
}

/// Requests sent from the LLM service to the inference engine.
pub enum EngineRequest {
    /// Generate a complete response.
    GenerateRequest(SequenceGroup, oneshot::Sender<GenerateRequestOutput>),
    /// Generate a streaming response.
    GenerateStreamingRequest(SequenceGroup, flume::Sender<StreamResponse>),
}
