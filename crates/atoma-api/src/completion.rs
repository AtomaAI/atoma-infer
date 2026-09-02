//! One chat completion's progress: the engine's events in, and what the client hears out.
//!
//! Every token event goes through the request's detokenizer, so the client hears text and never
//! a token; the engine's finish becomes the API's finish reason and usage, and a stop string
//! matched here finishes the request as a stop with the usage counted here, since the engine
//! never learns of the match beyond the cancel.

use atoma_core::request::{FinishReason as EngineFinish, RequestEvent};
use axum::http::StatusCode;
use thiserror::Error;

use crate::api::chat_completions::{
    ChatCompletionChunk, ChatCompletionResponse, FinishReason, Usage,
};
use crate::detokenize::{Detokenizer, Emission};

/// What one engine event moved the completion to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Progress {
    /// Nothing the client hears yet: the token completed no text.
    Nothing,
    /// New text.
    Text(String),
    /// The request is over with a completion: the last of its text, why it ended, and its usage.
    Finished {
        text: String,
        finish_reason: FinishReason,
        usage: Usage,
    },
    /// The request ended without a completion.
    Failed(Failed),
}

/// Why a request ended without a completion, and the status the client hears.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Failed {
    #[error(
        "the prompt is {prompt_tokens} tokens, which leaves nothing to generate under the max \
         model length of {max_model_length}"
    )]
    PromptTooLong {
        prompt_tokens: usize,
        max_model_length: usize,
    },
    #[error("the prompt has no tokens")]
    EmptyPrompt,
    #[error("the client left {queued} events unread, past the {max_backlog} allowed")]
    Backlogged { queued: usize, max_backlog: usize },
    #[error("the request was cancelled")]
    Cancelled,
    #[error("the engine shut down")]
    Shutdown,
    #[error("the executor was lost")]
    ExecutorLost,
}

impl Failed {
    /// The status a request that failed this way answers with.
    #[must_use]
    pub fn status(&self) -> StatusCode {
        match self {
            Self::PromptTooLong { .. } | Self::EmptyPrompt => StatusCode::BAD_REQUEST,
            Self::Backlogged { .. } | Self::Cancelled => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Shutdown | Self::ExecutorLost => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

/// One request on its way from the engine's events to the client's response.
pub struct Completion {
    id: String,
    model: String,
    created: u64,
    detokenizer: Detokenizer,
    prompt_tokens: usize,
    generated: usize,
}

impl Completion {
    /// A completion known to the client as `id`, over `prompt_tokens` of prompt.
    #[must_use]
    pub fn new(
        id: String,
        model: String,
        created: u64,
        detokenizer: Detokenizer,
        prompt_tokens: usize,
    ) -> Self {
        Self {
            id,
            model,
            created,
            detokenizer,
            prompt_tokens,
            generated: 0,
        }
    }

    /// Applies one engine event.
    ///
    /// # Errors
    ///
    /// Returns the tokenizer's error when a token cannot be decoded.
    pub fn apply(&mut self, event: RequestEvent) -> Result<Progress, tokenizers::Error> {
        match event {
            RequestEvent::Token { token, .. } => {
                self.generated += 1;
                Ok(match self.detokenizer.feed(token)? {
                    Emission::Text(text) if text.is_empty() => Progress::Nothing,
                    Emission::Text(text) => Progress::Text(text),
                    Emission::Stopped(text) => Progress::Finished {
                        text,
                        finish_reason: FinishReason::Stop,
                        usage: Usage::new(self.prompt_tokens, self.generated),
                    },
                })
            }
            RequestEvent::Finished { reason, usage, .. } => Ok(match completed(reason) {
                Ok(finish_reason) => Progress::Finished {
                    text: self.detokenizer.finish(),
                    finish_reason,
                    usage: Usage::from(usage),
                },
                Err(failed) => Progress::Failed(failed),
            }),
        }
    }

    /// A chunk of new text.
    #[must_use]
    pub fn chunk(&self, text: String) -> ChatCompletionChunk {
        ChatCompletionChunk::text(self.id.clone(), self.model.clone(), self.created, text)
    }

    /// The last chunk.
    #[must_use]
    pub fn last_chunk(&self, finish_reason: FinishReason, usage: Usage) -> ChatCompletionChunk {
        ChatCompletionChunk::finished(
            self.id.clone(),
            self.model.clone(),
            self.created,
            finish_reason,
            usage,
        )
    }

    /// The whole response, once finished.
    #[must_use]
    pub fn response(self, finish_reason: FinishReason, usage: Usage) -> ChatCompletionResponse {
        ChatCompletionResponse::completed(
            self.id,
            self.model,
            self.created,
            self.detokenizer.text().to_owned(),
            finish_reason,
            usage,
        )
    }
}

/// The API's finish reason for the engine's, or why there is none.
fn completed(reason: EngineFinish) -> Result<FinishReason, Failed> {
    match reason {
        EngineFinish::EndOfSequence => Ok(FinishReason::Stop),
        EngineFinish::MaxNewTokens | EngineFinish::MaxModelLength => Ok(FinishReason::Length),
        EngineFinish::PromptExceedsMaxModelLength {
            prompt_tokens,
            max_model_length,
        } => Err(Failed::PromptTooLong {
            prompt_tokens,
            max_model_length,
        }),
        EngineFinish::EmptyPrompt => Err(Failed::EmptyPrompt),
        EngineFinish::Cancelled => Err(Failed::Cancelled),
        EngineFinish::ClientBacklogged {
            queued,
            max_backlog,
        } => Err(Failed::Backlogged {
            queued,
            max_backlog,
        }),
        EngineFinish::Shutdown => Err(Failed::Shutdown),
        EngineFinish::ExecutorLost => Err(Failed::ExecutorLost),
    }
}

#[cfg(test)]
mod tests {
    use atoma_core::request::{FinishReason as EngineFinish, RequestEvent, Usage as EngineUsage};
    use atoma_core::types::{RequestId, SequenceIndex};
    use axum::http::StatusCode;

    use super::{Completion, Failed, Progress};
    use crate::api::chat_completions::{FinishReason, Usage};
    use crate::detokenize::Detokenizer;
    use crate::test_support::tokenizer;

    fn token(token: u32) -> RequestEvent {
        RequestEvent::Token {
            request: RequestId::new(1),
            sequence: SequenceIndex::new(0),
            token,
        }
    }

    fn finished(reason: EngineFinish, generated: usize) -> RequestEvent {
        RequestEvent::Finished {
            request: RequestId::new(1),
            reason,
            usage: EngineUsage {
                prompt_tokens: 3,
                generated_tokens: generated,
            },
        }
    }

    fn completion(stop: Vec<String>) -> Completion {
        let tokenizer = tokenizer();
        Completion::new(
            "chatcmpl-1".into(),
            "llama".into(),
            17,
            Detokenizer::new(tokenizer, stop),
            3,
        )
    }

    fn ids(text: &str) -> Vec<u32> {
        tokenizer().encode(text, false).unwrap().get_ids().to_vec()
    }

    #[test]
    fn tokens_become_text_and_the_engines_finish_becomes_the_apis() {
        let mut completion = completion(Vec::new());
        let mut text = String::new();
        for id in ids("hello") {
            match completion.apply(token(id)).unwrap() {
                Progress::Text(delta) => text.push_str(&delta),
                Progress::Nothing => {}
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(text, "hello");
        assert_eq!(
            completion
                .apply(finished(EngineFinish::EndOfSequence, 5))
                .unwrap(),
            Progress::Finished {
                text: String::new(),
                finish_reason: FinishReason::Stop,
                usage: Usage::new(3, 5),
            }
        );
        let response = completion.response(FinishReason::Stop, Usage::new(3, 5));
        assert_eq!(response.choices[0].finish_reason, FinishReason::Stop);
        assert_eq!(response.usage.total_tokens, 8);
        assert_eq!(response.created, 17);
        assert_eq!(response.id, "chatcmpl-1");
    }

    #[test]
    fn a_budget_reached_is_length_and_releases_the_held_back_tail() {
        let mut completion = completion(vec!["zzz".into()]);
        let mut text = String::new();
        for id in ids("hello") {
            if let Progress::Text(delta) = completion.apply(token(id)).unwrap() {
                text.push_str(&delta);
            }
        }
        assert_eq!(text, "hel", "two bytes are held back for the stop string");
        assert_eq!(
            completion
                .apply(finished(EngineFinish::MaxNewTokens, 5))
                .unwrap(),
            Progress::Finished {
                text: "lo".into(),
                finish_reason: FinishReason::Length,
                usage: Usage::new(3, 5),
            }
        );
    }

    #[test]
    fn a_stop_string_matched_here_finishes_as_a_stop_with_the_usage_counted_here() {
        let mut completion = completion(vec!["ll".into()]);
        let mut heard = String::new();
        let mut progress = None;
        for id in ids("hello") {
            match completion.apply(token(id)).unwrap() {
                finished @ Progress::Finished { .. } => {
                    progress = Some(finished);
                    break;
                }
                Progress::Text(text) => heard.push_str(&text),
                Progress::Nothing => {}
                Progress::Failed(failed) => panic!("{failed}"),
            }
        }
        let Some(Progress::Finished {
            text,
            finish_reason,
            usage,
        }) = progress
        else {
            panic!("the stop string was matched");
        };
        heard.push_str(&text);
        assert_eq!(heard, "he", "the text before the match, and no more");
        assert_eq!(finish_reason, FinishReason::Stop);
        assert_eq!(usage.prompt_tokens, 3);
        assert!(usage.completion_tokens >= 1);
        assert_eq!(
            completion.response(finish_reason, usage).choices[0].message,
            {
                use crate::api::chat_completions::{Message, MessageContent};
                Message::Assistant {
                    content: Some(MessageContent::Text("he".into())),
                    name: None,
                    refusal: None,
                    tool_calls: vec![],
                }
            }
        );
    }

    #[test]
    fn a_request_ending_without_a_completion_fails_with_its_status() {
        let cases = [
            (
                EngineFinish::PromptExceedsMaxModelLength {
                    prompt_tokens: 40,
                    max_model_length: 32,
                },
                StatusCode::BAD_REQUEST,
            ),
            (EngineFinish::EmptyPrompt, StatusCode::BAD_REQUEST),
            (EngineFinish::Shutdown, StatusCode::SERVICE_UNAVAILABLE),
            (EngineFinish::ExecutorLost, StatusCode::SERVICE_UNAVAILABLE),
            (
                EngineFinish::ClientBacklogged {
                    queued: 2000,
                    max_backlog: 1024,
                },
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (EngineFinish::Cancelled, StatusCode::INTERNAL_SERVER_ERROR),
        ];
        for (reason, status) in cases {
            let mut completion = completion(Vec::new());
            let Progress::Failed(failed) = completion.apply(finished(reason, 0)).unwrap() else {
                panic!("{reason:?} is not a completion");
            };
            assert_eq!(failed.status(), status, "{failed}");
        }
        assert!(Failed::PromptTooLong {
            prompt_tokens: 40,
            max_model_length: 32
        }
        .to_string()
        .contains("40 tokens"));
    }

    #[test]
    fn chunks_carry_the_completions_identity() {
        let completion = completion(Vec::new());
        let chunk = completion.chunk("hi".into());
        assert_eq!(chunk.id, "chatcmpl-1");
        assert_eq!(chunk.model, "llama");
        assert_eq!(chunk.created, 17);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hi"));
        let last = completion.last_chunk(FinishReason::Length, Usage::new(3, 4));
        assert_eq!(last.choices[0].finish_reason, Some(FinishReason::Length));
        assert_eq!(last.usage.total_tokens, 7);
    }
}
