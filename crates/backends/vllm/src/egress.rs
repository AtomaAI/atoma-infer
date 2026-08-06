//! Delivery of engine outputs back to the clients waiting for them.
//!
//! Every failure here is scoped to one request: a client that hung up costs that request its
//! channels, and nothing else. The engine step serves every other in-flight request in the same
//! batch, so it must never unwind on a broken client channel.

use std::collections::HashMap;

use tokio::sync::oneshot;
use tracing::{info, instrument};

use crate::output::{GenerateRequestOutput, StreamResponse};

/// Whether the client of a request is still listening.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientState {
    /// The client is still waiting for output, or the request has no client channel left to fail.
    Connected,
    /// The client's receiver is gone; the request can be dropped.
    Disconnected,
}

impl ClientState {
    /// Combines two observations of the same client: one failed channel is enough to call it gone.
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Connected, Self::Connected) => Self::Connected,
            (Self::Disconnected, _) | (_, Self::Disconnected) => Self::Disconnected,
        }
    }
}

/// The client channels of the requests currently in flight, keyed by request id.
#[derive(Debug, Default)]
pub struct ResponseSenders {
    /// Senders for non-streaming requests, which receive a single output when the request ends.
    completions: HashMap<String, oneshot::Sender<GenerateRequestOutput>>,
    /// Senders for streaming requests, which receive one message per generated token.
    streams: HashMap<String, flume::Sender<StreamResponse>>,
}

impl ResponseSenders {
    /// Registers the client of a non-streaming request.
    pub fn register_completion(
        &mut self,
        request_id: String,
        sender: oneshot::Sender<GenerateRequestOutput>,
    ) {
        self.completions.insert(request_id, sender);
    }

    /// Registers the client of a streaming request.
    pub fn register_stream(&mut self, request_id: String, sender: flume::Sender<StreamResponse>) {
        self.streams.insert(request_id, sender);
    }

    /// Streams one message to a request's client.
    ///
    /// Requests without a stream — non-streaming requests, and requests whose client has already
    /// been dropped — report `Connected`, since there is nothing left to fail.
    #[instrument(skip_all)]
    pub fn send_chunk(&self, request_id: &str, response: StreamResponse) -> ClientState {
        let Some(sender) = self.streams.get(request_id) else {
            return ClientState::Connected;
        };

        if sender.send(response).is_err() {
            info!("Client of streaming request {request_id} disconnected");
            return ClientState::Disconnected;
        }
        ClientState::Connected
    }

    /// Delivers a finished request's output and retires its client channels.
    ///
    /// A streaming request has no completion sender — its client was answered chunk by chunk — so
    /// for those this only retires the stream.
    #[instrument(skip_all)]
    pub fn complete(&mut self, output: GenerateRequestOutput) -> ClientState {
        let request_id = output.request_id.clone();
        self.streams.remove(&request_id);

        let Some(sender) = self.completions.remove(&request_id) else {
            return ClientState::Connected;
        };

        if sender.send(output).is_err() {
            info!("Client of request {request_id} disconnected before its output was sent");
            return ClientState::Disconnected;
        }
        ClientState::Connected
    }

    /// Whether the client of a non-streaming request is still waiting for its output.
    ///
    /// Nothing is sent to such a client until its request finishes, so its channel is probed
    /// instead: a request nobody is waiting for should not hold KV blocks for the rest of its
    /// generation.
    pub fn completion_client_state(&self, request_id: &str) -> ClientState {
        match self.completions.get(request_id) {
            Some(sender) if sender.is_closed() => ClientState::Disconnected,
            Some(_) | None => ClientState::Connected,
        }
    }

    /// Drops every client channel of a request, without sending anything.
    pub fn remove(&mut self, request_id: &str) {
        self.completions.remove(request_id);
        self.streams.remove(request_id);
    }

    /// Whether any request still holds a client channel.
    pub fn is_empty(&self) -> bool {
        self.completions.is_empty() && self.streams.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, RwLock},
        time::Instant,
    };

    use super::*;
    use crate::{output::GenerateStreamingOutput, sequence::RequestMetrics};

    fn request_output(request_id: &str) -> GenerateRequestOutput {
        GenerateRequestOutput {
            request_id: request_id.to_string(),
            prompt: "prompt".to_string(),
            inference_outputs: vec![],
            prompt_token_ids: vec![],
            is_finished: true,
            metrics: Arc::new(RwLock::new(RequestMetrics {
                arrival_time: Instant::now(),
                last_token_time: Instant::now(),
                first_scheduled_time: None,
                first_token_time: None,
                time_in_queue: None,
                finished_time: None,
            })),
        }
    }

    fn streaming_output(request_id: &str) -> StreamResponse {
        StreamResponse::Chunk(GenerateStreamingOutput {
            request_id: request_id.to_string(),
            created: 0,
            finish_reason: None,
            logprobs: vec![],
            num_prompt_tokens: 0,
            num_completion_tokens: 1,
            output_text: "token".to_string(),
        })
    }

    #[tokio::test]
    async fn test_completion_reaches_a_waiting_client() {
        let mut senders = ResponseSenders::default();
        let (sender, receiver) = oneshot::channel();
        senders.register_completion("request".to_string(), sender);

        assert_eq!(
            senders.complete(request_output("request")),
            ClientState::Connected
        );
        assert_eq!(
            receiver.await.expect("Output was not sent").request_id,
            "request"
        );
    }

    #[test]
    fn test_completion_to_a_dropped_client_is_reported() {
        let mut senders = ResponseSenders::default();
        let (sender, receiver) = oneshot::channel();
        senders.register_completion("request".to_string(), sender);
        drop(receiver);

        assert_eq!(
            senders.complete(request_output("request")),
            ClientState::Disconnected
        );
        assert!(senders.is_empty(), "a finished request keeps no channels");
    }

    #[test]
    fn test_completing_a_request_retires_its_stream() {
        let mut senders = ResponseSenders::default();
        let (sender, receiver) = flume::unbounded();
        senders.register_stream("request".to_string(), sender);

        assert_eq!(
            senders.complete(request_output("request")),
            ClientState::Connected
        );
        assert!(senders.is_empty());
        assert!(
            receiver.recv().is_err(),
            "the stream ends once the engine drops its sender"
        );
    }

    #[test]
    fn test_chunks_reach_a_streaming_client() {
        let mut senders = ResponseSenders::default();
        let (sender, receiver) = flume::unbounded();
        senders.register_stream("request".to_string(), sender);

        assert_eq!(
            senders.send_chunk("request", streaming_output("request")),
            ClientState::Connected
        );
        assert!(matches!(
            receiver.recv().expect("No chunk was streamed"),
            StreamResponse::Chunk(_)
        ));
    }

    #[test]
    fn test_chunk_to_a_dropped_client_is_reported() {
        let mut senders = ResponseSenders::default();
        let (sender, receiver) = flume::unbounded();
        senders.register_stream("request".to_string(), sender);
        drop(receiver);

        assert_eq!(
            senders.send_chunk("request", streaming_output("request")),
            ClientState::Disconnected
        );
    }

    /// The property the engine depends on: one client hanging up leaves every other client's
    /// channel intact.
    #[test]
    fn test_one_disconnected_client_does_not_disturb_the_others() {
        const NUM_REQUESTS: usize = 32;
        const DROPPED: usize = 7;

        let mut senders = ResponseSenders::default();
        let mut receivers = HashMap::new();
        for i in 0..NUM_REQUESTS {
            let (sender, receiver) = flume::unbounded();
            senders.register_stream(format!("request-{i}"), sender);
            if i == DROPPED {
                drop(receiver);
            } else {
                receivers.insert(format!("request-{i}"), receiver);
            }
        }

        let mut disconnected = Vec::new();
        for i in 0..NUM_REQUESTS {
            let request_id = format!("request-{i}");
            if senders.send_chunk(&request_id, streaming_output(&request_id))
                == ClientState::Disconnected
            {
                disconnected.push(request_id);
            }
        }

        assert_eq!(disconnected, vec![format!("request-{DROPPED}")]);
        for (request_id, receiver) in receivers.iter() {
            assert!(
                receiver.recv().is_ok(),
                "{request_id} lost its chunk to another request's disconnect"
            );
        }

        // Retiring the disconnected request leaves every other client streaming.
        for request_id in disconnected {
            senders.remove(&request_id);
        }
        for (request_id, receiver) in receivers.iter() {
            assert_eq!(
                senders.send_chunk(request_id, streaming_output(request_id)),
                ClientState::Connected
            );
            assert!(receiver.recv().is_ok());
        }
    }

    #[test]
    fn test_one_failed_channel_makes_the_client_disconnected() {
        assert_eq!(
            ClientState::Connected.and(ClientState::Connected),
            ClientState::Connected
        );
        assert_eq!(
            ClientState::Connected.and(ClientState::Disconnected),
            ClientState::Disconnected
        );
        assert_eq!(
            ClientState::Disconnected.and(ClientState::Connected),
            ClientState::Disconnected
        );
    }

    #[test]
    fn test_a_waiting_completion_client_is_reported_as_connected() {
        let mut senders = ResponseSenders::default();
        let (sender, _receiver) = oneshot::channel();
        senders.register_completion("request".to_string(), sender);

        assert_eq!(
            senders.completion_client_state("request"),
            ClientState::Connected
        );
    }

    /// A non-streaming client that hangs up mid-generation is visible before its output is ready,
    /// which is what lets the engine retire the request instead of generating for nobody.
    #[test]
    fn test_a_dropped_completion_client_is_reported_before_its_output_is_ready() {
        let mut senders = ResponseSenders::default();
        let (sender, receiver) = oneshot::channel();
        senders.register_completion("request".to_string(), sender);
        drop(receiver);

        assert_eq!(
            senders.completion_client_state("request"),
            ClientState::Disconnected
        );
    }

    #[test]
    fn test_an_unknown_request_has_no_completion_client_to_lose() {
        let senders = ResponseSenders::default();

        assert_eq!(
            senders.completion_client_state("request"),
            ClientState::Connected
        );
    }

    #[test]
    fn test_sending_to_an_unknown_request_is_a_no_op() {
        let mut senders = ResponseSenders::default();

        assert_eq!(
            senders.send_chunk("request", streaming_output("request")),
            ClientState::Connected
        );
        assert_eq!(
            senders.complete(request_output("request")),
            ClientState::Connected
        );
        assert!(senders.is_empty());
    }

    #[tokio::test]
    async fn test_remove_drops_both_channels_of_a_request() {
        let mut senders = ResponseSenders::default();
        let (completion_sender, completion_receiver) = oneshot::channel();
        let (stream_sender, stream_receiver) = flume::unbounded();
        senders.register_completion("request".to_string(), completion_sender);
        senders.register_stream("request".to_string(), stream_sender);
        assert!(!senders.is_empty());

        senders.remove("request");

        assert!(senders.is_empty());
        assert!(stream_receiver.recv().is_err());
        assert!(completion_receiver.await.is_err());
    }
}
