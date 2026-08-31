//! What a request's client hears: incremental token events and one finish.

use serde::{Deserialize, Serialize};

use crate::types::{RequestId, SequenceIndex};

/// One message on a request's egress channel. Incremental, never cumulative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestEvent {
    /// One token sampled for `sequence`, in generation order.
    Token {
        request: RequestId,
        sequence: SequenceIndex,
        token: u32,
    },
    /// The request is over. Nothing follows on the channel.
    Finished {
        request: RequestId,
        reason: FinishReason,
        usage: Usage,
    },
}

/// Why a request finished. Every reason that rejects a request carries the numbers behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model emitted an end-of-sequence token.
    EndOfSequence,
    /// The request generated everything it asked for.
    MaxNewTokens,
    /// The sequence reached the longest length the model serves.
    MaxModelLength,
    /// The prompt alone leaves no room to generate under the max model length.
    PromptExceedsMaxModelLength {
        prompt_tokens: usize,
        max_model_length: usize,
    },
    /// The prompt has no tokens, so there is nothing to compute a first token from.
    EmptyPrompt,
    /// The client dropped its egress receiver.
    Cancelled,
    /// The engine shut down with the request live.
    Shutdown,
    /// The executor went away with the request live.
    ExecutorLost,
}

/// Token accounting reported with a finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::{FinishReason, RequestEvent, Usage};
    use crate::types::{RequestId, SequenceIndex};

    #[test]
    fn events_round_trip_through_serde() {
        let events = [
            RequestEvent::Token {
                request: RequestId::new(3),
                sequence: SequenceIndex::new(0),
                token: 42,
            },
            RequestEvent::Finished {
                request: RequestId::new(3),
                reason: FinishReason::PromptExceedsMaxModelLength {
                    prompt_tokens: 40,
                    max_model_length: 32,
                },
                usage: Usage {
                    prompt_tokens: 40,
                    generated_tokens: 0,
                },
            },
        ];
        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            assert_eq!(serde_json::from_str::<RequestEvent>(&json).unwrap(), event);
        }
    }

    #[test]
    fn finish_reasons_serialize_as_snake_case_tags() {
        let json = serde_json::to_string(&FinishReason::EndOfSequence).unwrap();
        assert_eq!(json, "\"end_of_sequence\"");
    }
}
