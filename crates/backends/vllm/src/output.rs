use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Instant,
};

use tracing::debug;

use crate::sequence::{LogProb, RequestMetrics, SequenceGroup};

/// Output of running AI inference over a `SequenceGroup`.
#[derive(Debug)]
pub struct GenerateRequestOutput {
    /// Request id
    pub request_id: String,
    /// The `String` prompt
    pub prompt: String,
    /// Inference outputs
    pub inference_outputs: Vec<InferenceOutput>,
    /// Prompt token ids
    pub prompt_token_ids: Vec<u32>,
    /// Is finished
    pub is_finished: bool,
    /// Metrics
    pub metrics: Arc<RwLock<RequestMetrics>>,
}

impl GenerateRequestOutput {
    /// Creates a new `Self` instance from a `SequenceGroup`.
    pub fn from_sequence_group(sequence_group: &SequenceGroup) -> Self {
        debug!(
            "Creating `GenerateRequestOutput` from sequence group with id = {}",
            sequence_group.request_id
        );
        let mut sequences = sequence_group.sequences.values().collect::<Vec<_>>();

        let top_n_sequences = if sequences.len() == 1 {
            sequences
        } else {
            let n = sequence_group.next_token_chooser_params().n;
            sequences.sort_by(|s1, s2| {
                s1.read()
                    .unwrap()
                    .cumulative_logprob()
                    .partial_cmp(&s2.read().unwrap().cumulative_logprob())
                    .unwrap()
            });
            sequences[..n].to_vec()
        };

        let inference_outputs = top_n_sequences
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let s = s.read().unwrap();
                InferenceOutput {
                    index: i,
                    output_text: s.get_output_text(),
                    token_ids: s.get_token_ids(),
                    cumulative_logprob: s.cumulative_logprob(),
                    logprobs: s.output_logprobs.clone(),
                    finish_reason: s.get_sequence_status().finished_reason(),
                    stop_reason: s.stop_reason,
                }
            })
            .collect::<Vec<_>>();

        let is_finished = sequence_group.is_finished();
        if is_finished {
            sequence_group.set_finished_time(Instant::now());
        }
        Self {
            request_id: sequence_group.request_id.clone(),
            inference_outputs,
            prompt: sequence_group.prompt(),
            prompt_token_ids: sequence_group.prompt_token_ids(),
            is_finished,
            metrics: sequence_group.metrics.clone(),
        }
    }
}

/// Output of running AI inference on one sequence in a sequence group.
#[derive(Clone, Debug)]
pub struct InferenceOutput {
    /// The index of the output in the request
    pub index: usize,
    /// The generated output text
    pub output_text: String,
    /// The token ids of the generated output text
    pub token_ids: Vec<u32>,
    /// The cumulative log probability of the generated output text
    pub cumulative_logprob: f32,
    /// The log probabilities of the top probability words at each position
    pub logprobs: Vec<HashMap<u32, LogProb>>,
    /// The reason why the sequence is finished
    pub finish_reason: Option<String>,
    /// The stop token id, if one caused completion
    pub stop_reason: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct GenerateStreamingOutput {
    pub request_id: String,
    pub created: u64,
    pub finish_reason: Option<String>,
    pub logprobs: Vec<HashMap<u32, LogProb>>,
    pub num_prompt_tokens: usize,
    pub num_completion_tokens: usize,
    pub output_text: String,
}

/// Responses emitted while streaming a generation.
pub enum StreamResponse {
    /// A generated output chunk is available.
    Chunk(GenerateStreamingOutput),
    /// Generation failed.
    Error(String),
    /// Generation completed successfully.
    Finished,
}
