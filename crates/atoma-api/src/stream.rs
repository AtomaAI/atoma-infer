//! Streaming one request's completion as server-sent events: a chunk per piece of new text, the
//! last chunk with the finish and the usage, then `[DONE]`.
//!
//! The engine's events are polled through a stream rather than a bare receiver so that a poll on
//! an empty channel registers a waker: an SSE response that returns `Pending` without one is never
//! polled again, and the client hangs until it gives up. The receiver is dropped the moment the
//! request is over, which is the cancel a stop string matched here needs.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use atoma_core::request::{EgressReceiver, RequestEvent};
use axum::response::sse::Event;
use axum::Error;
use flume::r#async::RecvStream;
use futures::stream::Stream;
use serde_json::json;

use crate::completion::{Completion, Progress};

/// What the client is told when the engine closes the request without a finish.
const CLOSED_WITHOUT_FINISH: &str = "the engine closed the request without a finish";

/// One streamed chat completion.
pub struct Streamer {
    /// The engine's events for this request; dropped once the request is over.
    events: Option<RecvStream<'static, RequestEvent>>,
    completion: Completion,
    /// Events ready to go out: one engine event can yield several.
    pending: VecDeque<Result<Event, Error>>,
}

impl Streamer {
    #[must_use]
    pub fn new(receiver: EgressReceiver, completion: Completion) -> Self {
        Self {
            events: Some(receiver.into_stream()),
            completion,
            pending: VecDeque::new(),
        }
    }

    /// Queues `event` and, when it ends the request, drops the engine's channel.
    fn handle(&mut self, event: RequestEvent) {
        match self.completion.apply(event) {
            Ok(Progress::Nothing) => {}
            Ok(Progress::Text(text)) => self.queue_text(text),
            Ok(Progress::Finished {
                text,
                finish_reason,
                usage,
            }) => {
                self.queue_text(text);
                self.pending.push_back(
                    Event::default().json_data(self.completion.last_chunk(finish_reason, usage)),
                );
                self.pending.push_back(Ok(Event::default().data("[DONE]")));
                self.events = None;
            }
            Ok(Progress::Failed(failed)) => {
                self.queue_error(&failed.to_string(), failed.status().as_u16());
            }
            Err(error) => {
                self.queue_error(&format!("a token cannot be decoded: {error}"), 500);
            }
        }
    }

    fn queue_text(&mut self, text: String) {
        if text.is_empty() {
            return;
        }
        self.pending
            .push_back(Event::default().json_data(self.completion.chunk(text)));
    }

    /// Queues an error event and ends the request.
    fn queue_error(&mut self, message: &str, status: u16) {
        self.pending.push_back(
            Event::default()
                .json_data(json!({ "error": { "message": message, "status": status } })),
        );
        self.events = None;
    }
}

impl Stream for Streamer {
    type Item = Result<Event, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Poll::Ready(Some(event));
            }
            let Some(events) = self.events.as_mut() else {
                return Poll::Ready(None);
            };
            match Pin::new(events).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(event)) => self.handle(event),
                Poll::Ready(None) => self.queue_error(CLOSED_WITHOUT_FINISH, 500),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use atoma_core::request::{egress, EgressSender, FinishReason, RequestEvent, Usage};
    use atoma_core::types::{RequestId, SequenceIndex};
    use futures::task::{waker, ArcWake};
    use futures::StreamExt;

    use super::*;
    use crate::api::chat_completions::CompletionIdentity;
    use crate::detokenize::Detokenizer;
    use crate::test_support::tokenizer;

    /// A waker that counts how many times the stream asked to be polled again.
    struct CountingWaker {
        wakes: AtomicUsize,
    }

    impl ArcWake for CountingWaker {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.wakes.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn streamer(stop: Vec<String>) -> (EgressSender, Streamer) {
        let (sender, receiver) = egress();
        let identity = CompletionIdentity {
            id: "chatcmpl-1".into(),
            model: "llama".into(),
            created: 17,
        };
        let completion = Completion::new(identity, Detokenizer::new(tokenizer(), stop), 3);
        (sender, Streamer::new(receiver, completion))
    }

    fn token_events(text: &str) -> Vec<RequestEvent> {
        tokenizer()
            .encode(text, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&token| RequestEvent::Token {
                request: RequestId::new(1),
                sequence: SequenceIndex::new(0),
                token,
            })
            .collect()
    }

    fn finished(reason: FinishReason, generated: usize) -> RequestEvent {
        RequestEvent::Finished {
            request: RequestId::new(1),
            reason,
            usage: Usage {
                prompt_tokens: 3,
                generated_tokens: generated,
            },
        }
    }

    /// The bytes an event will put on the wire. Axum keeps them inside the event, and its
    /// `Debug` output is the only view of them from outside that crate.
    fn wire(event: Event) -> String {
        format!("{event:?}")
    }

    async fn drain(streamer: Streamer) -> Vec<String> {
        streamer
            .map(|event| wire(event.expect("the streamer yielded an error")))
            .collect()
            .await
    }

    #[test]
    fn polling_an_empty_stream_registers_a_waker() {
        let (sender, mut streamer) = streamer(Vec::new());
        let counting = Arc::new(CountingWaker {
            wakes: AtomicUsize::new(0),
        });
        let waker = waker(Arc::clone(&counting));
        let mut context = Context::from_waker(&waker);

        assert!(Pin::new(&mut streamer).poll_next(&mut context).is_pending());
        assert_eq!(counting.wakes.load(Ordering::SeqCst), 0);
        for event in token_events("hi") {
            sender.send(event);
        }
        assert!(
            counting.wakes.load(Ordering::SeqCst) >= 1,
            "an event arriving must wake the task polling the stream"
        );
        assert!(Pin::new(&mut streamer).poll_next(&mut context).is_ready());
    }

    #[tokio::test]
    async fn text_chunks_are_followed_by_the_finish_and_done() {
        let (sender, streamer) = streamer(Vec::new());
        let tokens = token_events("hello");
        let generated = tokens.len();
        for event in tokens {
            sender.send(event);
        }
        sender.send(finished(FinishReason::MaxNewTokens, generated));

        let events = drain(streamer).await;
        assert!(events.len() >= 3, "{events:?}");
        assert!(events[0].contains("chat.completion.chunk"), "{}", events[0]);
        let last_chunk = &events[events.len() - 2];
        assert!(
            last_chunk.contains("\\\"finish_reason\\\":\\\"length\\\""),
            "{last_chunk}"
        );
        assert!(
            last_chunk.contains(&format!("\\\"total_tokens\\\":{}", 3 + generated)),
            "{last_chunk}"
        );
        assert!(events.last().unwrap().contains("[DONE]"));
    }

    #[tokio::test]
    async fn a_stop_string_cancels_the_request_and_ends_the_stream_as_a_stop() {
        let (sender, mut streamer) = streamer(vec!["ll".into()]);
        for event in token_events("hello") {
            sender.send(event);
        }
        let mut events = Vec::new();
        while let Some(event) = streamer.next().await {
            events.push(wire(event.unwrap()));
        }
        assert!(
            sender.is_cancelled(),
            "the receiver was dropped at the match, which is the cancel"
        );
        let last_chunk = &events[events.len() - 2];
        assert!(
            last_chunk.contains("\\\"finish_reason\\\":\\\"stop\\\""),
            "{last_chunk}"
        );
        assert!(events.last().unwrap().contains("[DONE]"));
        assert_eq!(events.len(), 4, "h, e, the finish and DONE: {events:?}");
        assert!(
            events[0].contains("\\\"content\\\":\\\"h\\\""),
            "{}",
            events[0]
        );
        assert!(
            events[1].contains("\\\"content\\\":\\\"e\\\""),
            "{}",
            events[1]
        );
    }

    #[tokio::test]
    async fn a_request_that_fails_ends_the_stream_with_an_error_event() {
        let (sender, streamer) = streamer(Vec::new());
        sender.send(finished(FinishReason::ExecutorLost, 0));
        let events = drain(streamer).await;
        assert_eq!(events.len(), 1, "{events:?}");
        assert!(events[0].contains("executor was lost"), "{}", events[0]);
        assert!(events[0].contains("503"), "{}", events[0]);
    }

    #[tokio::test]
    async fn a_channel_closed_without_a_finish_ends_the_stream_with_an_error_event() {
        let (sender, streamer) = streamer(Vec::new());
        for event in token_events("hi") {
            sender.send(event);
        }
        drop(sender);
        let events = drain(streamer).await;
        assert!(
            events.last().unwrap().contains(CLOSED_WITHOUT_FINISH),
            "{events:?}"
        );
    }
}
