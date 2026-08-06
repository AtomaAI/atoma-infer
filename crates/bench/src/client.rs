//! The load generator's client.
//!
//! Sends one streaming chat completion per request and times what comes back: the first content
//! token (TTFT), the gap to each following token (ITL), and the end of the response. Both engines
//! are driven through the same client over the same OpenAI surface, so the numbers differ by
//! engine and not by client.

use std::time::{Duration, Instant};

use futures::StreamExt;

use crate::{
    error::{BenchError, Result},
    record::RequestRecord,
    workload::BenchRequest,
};

/// How much of an engine's error body is kept in a failure record.
const MAX_ERROR_BODY_CHARS: usize = 200;

/// What one decoded server-sent event carried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SseEvent {
    /// A generated token.
    Delta(String),
    /// The stream ended.
    Done,
    /// The engine sent something the harness cannot read.
    Malformed(String),
}

/// Decodes an OpenAI streaming response out of the byte chunks it arrives in.
///
/// Events are buffered until complete: a chunk boundary in the middle of an event is normal, and
/// parsing each read on its own would drop tokens.
#[derive(Debug, Default)]
pub struct SseDecoder {
    /// Bytes received but not yet part of a complete event.
    buffer: String,
}

impl SseDecoder {
    /// Feeds a chunk of the response body in and takes out whatever events it completed.
    pub fn push(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        while let Some(boundary) = self.buffer.find("\n\n") {
            let event = self.buffer[..boundary].to_string();
            self.buffer.drain(..boundary + 2);
            events.extend(decode_event(&event));
        }

        events
    }
}

/// Decodes one complete server-sent event.
///
/// Yields nothing for keep-alive comments, for the role-only opening chunk, and for chunks whose
/// content is empty: none of them is a token, and counting them would move TTFT a decode step
/// earlier than the engine managed.
fn decode_event(event: &str) -> Option<SseEvent> {
    let payload = event
        .lines()
        .find_map(|line| line.strip_prefix("data:"))?
        .trim();

    if payload == "[DONE]" {
        return Some(SseEvent::Done);
    }

    let chunk: serde_json::Value = match serde_json::from_str(payload) {
        Ok(chunk) => chunk,
        Err(error) => return Some(SseEvent::Malformed(format!("{error}: {payload}"))),
    };

    match chunk["choices"][0]["delta"]["content"].as_str() {
        Some("") | None => None,
        Some(content) => Some(SseEvent::Delta(content.to_string())),
    }
}

/// The body of one benchmark completion.
///
/// `max_completion_tokens` rather than the deprecated `max_tokens`: both engines accept it, and an
/// engine that ignored an unknown length field would generate until its own stop condition and
/// answer a different question than the one being measured. Temperature is zero so the sampler
/// contributes as little variance as it can to the comparison.
fn completion_body(model: &str, request: &BenchRequest) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": request.prompt }],
        "max_completion_tokens": request.max_tokens,
        "stream": true,
        "temperature": 0.0,
    })
}

/// Drives one engine's OpenAI surface.
pub struct OpenAiClient {
    /// The chat completions endpoint.
    completions_url: String,
    /// The model id sent with every request.
    model: String,
    /// Optional bearer token.
    api_key: Option<String>,
    /// The HTTP client, whose connection pool is shared across the run's requests.
    http: reqwest::Client,
}

impl OpenAiClient {
    /// Builds a client against `base_url` (for example `http://127.0.0.1:8080`).
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Config`] if the client cannot be built.
    pub fn new(
        base_url: &str,
        model: String,
        api_key: Option<String>,
        timeout: Duration,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|error| {
                BenchError::Config(format!("Failed to build the HTTP client: {error}"))
            })?;

        Ok(Self {
            completions_url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')),
            model,
            api_key,
            http,
        })
    }

    /// Sends one request and records what the engine streamed back.
    ///
    /// Transport failures, error statuses and unreadable streams are recorded as failed requests
    /// rather than raised: a run measures how an engine behaves under load, and refusing load is
    /// part of that behaviour.
    pub async fn stream_completion(&self, request: &BenchRequest) -> RequestRecord {
        let started = Instant::now();
        let response = self
            .send(request)
            .await
            .and_then(|response| response.error_for_status());

        let response = match response {
            Ok(response) => response,
            Err(error) => return self.record_failure(request, &error.to_string(), started),
        };

        self.record_stream(request, response, started).await
    }

    /// Posts the completion request.
    async fn send(&self, request: &BenchRequest) -> reqwest::Result<reqwest::Response> {
        let post = self
            .http
            .post(&self.completions_url)
            .json(&completion_body(&self.model, request));
        let post = match &self.api_key {
            Some(api_key) => post.bearer_auth(api_key),
            None => post,
        };

        post.send().await
    }

    /// Times the streamed tokens of an accepted request.
    async fn record_stream(
        &self,
        request: &BenchRequest,
        response: reqwest::Response,
        started: Instant,
    ) -> RequestRecord {
        let mut decoder = SseDecoder::default();
        let mut token_offsets = Vec::new();
        let mut body = response.bytes_stream();

        while let Some(chunk) = body.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    return self.record_failure(request, &error.to_string(), started);
                }
            };

            for event in decoder.push(&String::from_utf8_lossy(&chunk)) {
                match event {
                    SseEvent::Delta(_) => token_offsets.push(started.elapsed()),
                    SseEvent::Done => {
                        return RequestRecord::completed(
                            request.id.clone(),
                            request.prompt_tokens,
                            &token_offsets,
                            started.elapsed(),
                        )
                    }
                    SseEvent::Malformed(reason) => {
                        return self.record_failure(
                            request,
                            &format!("the engine streamed an unreadable event: {reason}"),
                            started,
                        )
                    }
                }
            }
        }

        // The body ended without `[DONE]`: the engine hung up mid-answer.
        self.record_failure(
            request,
            &format!(
                "the stream ended after {} tokens without a completion event",
                token_offsets.len()
            ),
            started,
        )
    }

    /// Records a request the engine did not answer.
    fn record_failure(
        &self,
        request: &BenchRequest,
        reason: &str,
        started: Instant,
    ) -> RequestRecord {
        let reason = reason
            .chars()
            .take(MAX_ERROR_BODY_CHARS)
            .collect::<String>();
        RequestRecord::failed(
            request.id.clone(),
            request.prompt_tokens,
            reason,
            started.elapsed(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta(content: &str) -> String {
        format!(
            "data: {}\n\n",
            serde_json::json!({ "choices": [{ "delta": { "content": content } }] })
        )
    }

    fn decoded(chunks: &[&str]) -> Vec<SseEvent> {
        let mut decoder = SseDecoder::default();
        chunks
            .iter()
            .flat_map(|chunk| decoder.push(chunk))
            .collect()
    }

    #[test]
    fn test_content_deltas_are_decoded_in_order() {
        let events = decoded(&[&delta("Hel"), &delta("lo"), "data: [DONE]\n\n"]);

        assert_eq!(
            events,
            vec![
                SseEvent::Delta("Hel".to_string()),
                SseEvent::Delta("lo".to_string()),
                SseEvent::Done,
            ]
        );
    }

    /// An SSE event is split across TCP reads whenever it feels like it; a decoder that parses
    /// each read on its own would drop tokens and report a shorter, faster stream.
    #[test]
    fn test_an_event_split_across_reads_is_decoded_once_it_completes() {
        let whole = delta("token");
        let (head, tail) = whole.split_at(whole.len() / 2);

        let mut decoder = SseDecoder::default();
        assert!(decoder.push(head).is_empty());
        assert_eq!(
            decoder.push(tail),
            vec![SseEvent::Delta("token".to_string())]
        );
    }

    /// The opening chunk of an OpenAI stream announces the role and carries no content; counting
    /// it as a token would report a TTFT one decode step early.
    #[test]
    fn test_chunks_without_content_are_not_tokens() {
        let opening = format!(
            "data: {}\n\n",
            serde_json::json!({ "choices": [{ "delta": { "role": "assistant" } }] })
        );
        let empty = delta("");

        assert!(decoded(&[&opening, &empty]).is_empty());
    }

    #[test]
    fn test_comments_and_blank_lines_are_ignored() {
        let events = decoded(&[": keep-alive\n\n", "\n\n", &delta("x")]);

        assert_eq!(events, vec![SseEvent::Delta("x".to_string())]);
    }

    /// A stream the harness cannot parse is an engine defect, and the request it belongs to has to
    /// be reported as failed rather than quietly truncated.
    #[test]
    fn test_a_malformed_payload_is_surfaced() {
        let events = decoded(&["data: {not json}\n\n"]);

        assert!(
            matches!(events.as_slice(), [SseEvent::Malformed(_)]),
            "{events:?}"
        );
    }

    #[test]
    fn test_the_request_body_asks_for_a_bounded_streaming_completion() {
        let request = BenchRequest {
            id: "request-0".to_string(),
            prompt: "why is the sky blue".to_string(),
            prompt_tokens: 5,
            max_tokens: 128,
            arrival: Duration::ZERO,
        };

        let body = completion_body("meta-llama/Meta-Llama-3-8B-Instruct", &request);

        assert_eq!(body["model"], "meta-llama/Meta-Llama-3-8B-Instruct");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "why is the sky blue");
        assert_eq!(body["max_completion_tokens"], 128);
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0.0);
    }
}
