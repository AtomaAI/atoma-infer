use crate::api::chat_completions::ChatCompletionChunk;
use atoma_backends::StreamResponse;
use axum::{response::sse::Event, Error};
use flume::{r#async::RecvStream, Receiver};
use futures::stream::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A structure for streaming chat completion chunks.
///
/// `Streamer` manages the reception of `ChatCompletionChunk`s and tracks the current status
/// of the streaming process.
pub struct Streamer {
    /// The engine's messages for this request.
    ///
    /// This is a stream rather than a bare receiver so that a poll on an empty channel registers a
    /// waker: an SSE response that returns `Pending` without one is never polled again, and the
    /// client hangs until it gives up.
    responses: RecvStream<'static, StreamResponse>,
    /// The current status of the streaming process.
    status: StreamStatus,
    /// The model used for generating the output.
    model: String,
}

impl Streamer {
    /// Creates a new `Streamer` with the specified receiver and model.
    pub fn new(receiver: Receiver<StreamResponse>, model: String) -> Self {
        Self {
            responses: receiver.into_stream(),
            status: StreamStatus::NotStarted,
            model,
        }
    }
}

/// Represents the various states of a streaming process.
///
/// This enum is used to track and communicate the current state of a `Streamer`,
/// allowing for proper handling of different scenarios during streaming.
#[derive(Debug, PartialEq, Eq)]
pub enum StreamStatus {
    /// Indicates that the streaming process has not started yet.
    NotStarted,
    /// Indicates that the streaming process has started and is actively receiving data.
    ///
    /// This is the initial state when a stream begins and is ready to process incoming chunks.
    Started,
    /// Indicates that the streaming process has completed successfully.
    ///
    /// This state is reached when all data has been received and processed without errors.
    Completed,
    /// Indicates that the streaming process has failed, with an associated error message.
    ///
    /// This state is used when an error occurs during streaming, providing context about the
    /// failure.
    Failed {
        /// A descriptive error message explaining the reason for the failure.
        error: String,
    },
    /// Indicates that the streaming process was interrupted before completion.
    ///
    /// This state is used when the stream is stopped prematurely, either by user action or system
    /// events.
    Interrupted {
        /// A description of why the stream was interrupted.
        reason: String,
    },
}

impl Stream for Streamer {
    type Item = Result<Event, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.status == StreamStatus::Completed {
            return Poll::Ready(None);
        }

        match Pin::new(&mut self.responses).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                // The engine dropped the request's sender without finishing it.
                if self.status == StreamStatus::Started {
                    self.status = StreamStatus::Interrupted {
                        reason: "Stream disconnected".to_string(),
                    };
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(StreamResponse::Chunk(chunk))) => {
                if self.status != StreamStatus::Started {
                    self.status = StreamStatus::Started;
                }
                let response = ChatCompletionChunk::try_from((self.model.clone(), chunk))
                    .map_err(Error::new)?;
                Poll::Ready(Some(Event::default().json_data(response)))
            }
            Poll::Ready(Some(StreamResponse::Finished)) => {
                self.status = StreamStatus::Completed;
                Poll::Ready(Some(Ok(Event::default().data("[DONE]"))))
            }
            Poll::Ready(Some(StreamResponse::Error(error))) => {
                self.status = StreamStatus::Failed {
                    error: error.clone(),
                };
                Poll::Ready(Some(Ok(Event::default().data(error))))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use atoma_backends::GenerateStreamingOutput;
    use futures::{
        task::{waker, ArcWake},
        StreamExt,
    };

    use super::*;

    /// A waker that counts how many times the stream asked to be polled again.
    struct CountingWaker {
        wakes: AtomicUsize,
    }

    impl ArcWake for CountingWaker {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.wakes.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn chunk(output_text: &str) -> StreamResponse {
        StreamResponse::Chunk(GenerateStreamingOutput {
            request_id: "request".to_string(),
            created: 0,
            finish_reason: None,
            logprobs: vec![],
            num_prompt_tokens: 4,
            num_completion_tokens: 1,
            output_text: output_text.to_string(),
        })
    }

    /// The bytes an event will put on the wire. Axum keeps them inside the event, and its `Debug`
    /// output is the only view of them from outside that crate.
    fn event_data(event: Event) -> String {
        format!("{event:?}")
    }

    #[test]
    fn test_polling_an_empty_stream_registers_a_waker() {
        let (sender, receiver) = flume::unbounded();
        let mut streamer = Streamer::new(receiver, "model".to_string());
        let counting_waker = Arc::new(CountingWaker {
            wakes: AtomicUsize::new(0),
        });
        let waker = waker(counting_waker.clone());
        let mut context = Context::from_waker(&waker);

        assert!(Pin::new(&mut streamer).poll_next(&mut context).is_pending());
        assert_eq!(counting_waker.wakes.load(Ordering::SeqCst), 0);

        sender.send(chunk("hello")).expect("Failed to send chunk");

        assert_eq!(
            counting_waker.wakes.load(Ordering::SeqCst),
            1,
            "a chunk arriving must wake the task polling the stream"
        );
        assert!(Pin::new(&mut streamer).poll_next(&mut context).is_ready());
    }

    #[tokio::test]
    async fn test_chunks_are_streamed_until_the_finished_marker() {
        let (sender, receiver) = flume::unbounded();
        let streamer = Streamer::new(receiver, "model".to_string());

        sender.send(chunk("hello")).expect("Failed to send chunk");
        sender.send(chunk(" world")).expect("Failed to send chunk");
        sender
            .send(StreamResponse::Finished)
            .expect("Failed to send finish marker");

        let events = streamer
            .map(|event| event_data(event.expect("Streamer yielded an error")))
            .collect::<Vec<_>>()
            .await;

        assert_eq!(events.len(), 3);
        assert!(events[0].contains("hello"));
        assert!(events[1].contains(" world"));
        assert!(events[2].contains("[DONE]"));
    }

    /// The engine drops a request's sender when it retires the request, which has to end the SSE
    /// response rather than leave the client waiting.
    #[tokio::test]
    async fn test_stream_ends_when_the_engine_drops_the_sender() {
        let (sender, receiver) = flume::unbounded();
        let mut streamer = Streamer::new(receiver, "model".to_string());

        sender.send(chunk("hello")).expect("Failed to send chunk");
        drop(sender);

        assert!(streamer.next().await.is_some());
        assert!(streamer.next().await.is_none());
        assert_eq!(
            streamer.status,
            StreamStatus::Interrupted {
                reason: "Stream disconnected".to_string()
            }
        );
    }

    #[tokio::test]
    async fn test_engine_errors_are_forwarded_to_the_client() {
        let (sender, receiver) = flume::unbounded();
        let mut streamer = Streamer::new(receiver, "model".to_string());

        sender
            .send(StreamResponse::Error("model failed".to_string()))
            .expect("Failed to send error");

        let event = streamer
            .next()
            .await
            .expect("Streamer ended without forwarding the error")
            .expect("Streamer yielded an error");
        assert!(event_data(event).contains("model failed"));
        assert_eq!(
            streamer.status,
            StreamStatus::Failed {
                error: "model failed".to_string()
            }
        );
    }
}
