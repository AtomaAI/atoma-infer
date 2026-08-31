//! The per-request egress channel.
//!
//! The engine's side sends and gets nothing back: there is no error to ignore, so a client hanging
//! up can never unwind the engine. The client's side is the receiver, and dropping it is the
//! request's cancel, observed by the engine through [`EgressSender::is_cancelled`].
//!
//! A send is never refused, so a client's token stream can never gain a hole: it ends where
//! generation ended, always with a finish. A client that stops reading is bounded instead by the
//! scheduler, which retires a request whose [`EgressSender::backlog`] runs past its limit.

use flume::{Receiver, SendError, Sender};

use crate::request::RequestEvent;

/// The client's end of one request's egress channel. Dropping it cancels the request.
pub type EgressReceiver = Receiver<RequestEvent>;

/// Where a request's output goes.
///
/// A padding dummy is the one request with no client, so a request without a channel has exactly
/// one meaning and this names it rather than leaving it as an absence to explain.
#[derive(Debug)]
pub(crate) enum Egress {
    /// The channel to the client that submitted the request.
    Client(EgressSender),
    /// A padding dummy: no client to send to, to lose, or to fall behind.
    Dummy,
}

/// The engine's end of one request's egress channel.
///
/// The channel is unbounded on purpose: the engine must never block on a client, and a refused
/// send would leave a hole in the token stream. What bounds it is [`EgressSender::backlog`],
/// which the scheduler reads to retire a client that has stopped reading.
#[derive(Debug, Clone)]
pub struct EgressSender {
    sender: Sender<RequestEvent>,
}

impl EgressSender {
    /// Sends `event` to the client, returning nothing. A client that is gone is observed through
    /// [`EgressSender::is_cancelled`], not here.
    pub fn send(&self, event: RequestEvent) {
        // The only failure is a dropped receiver, which is the cancel signal read elsewhere.
        match self.sender.send(event) {
            Ok(()) | Err(SendError(_)) => {}
        }
    }

    /// Whether the client dropped its receiver — the one and only cancel.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.sender.is_disconnected()
    }

    /// Events the client has not read yet. A client that keeps up holds none.
    #[must_use]
    pub fn backlog(&self) -> usize {
        self.sender.len()
    }
}

/// Opens one request's egress channel: the engine keeps the sender, the client the receiver.
#[must_use]
pub fn egress() -> (EgressSender, EgressReceiver) {
    let (sender, receiver) = flume::unbounded();
    (EgressSender { sender }, receiver)
}

#[cfg(test)]
mod tests {
    use super::egress;
    use crate::request::{FinishReason, RequestEvent, Usage};
    use crate::types::{RequestId, SequenceIndex};

    fn token(token: u32) -> RequestEvent {
        RequestEvent::Token {
            request: RequestId::new(1),
            sequence: SequenceIndex::new(0),
            token,
        }
    }

    #[test]
    fn events_reach_a_listening_client_in_order() {
        let (sender, receiver) = egress();
        sender.send(token(7));
        sender.send(RequestEvent::Finished {
            request: RequestId::new(1),
            reason: FinishReason::EndOfSequence,
            usage: Usage {
                prompt_tokens: 4,
                generated_tokens: 1,
            },
        });
        assert_eq!(receiver.recv().unwrap(), token(7));
        assert!(matches!(
            receiver.recv().unwrap(),
            RequestEvent::Finished {
                reason: FinishReason::EndOfSequence,
                ..
            }
        ));
        assert!(!sender.is_cancelled());
    }

    #[test]
    fn a_dropped_receiver_is_the_cancel_and_later_sends_return_nothing() {
        let (sender, receiver) = egress();
        assert!(!sender.is_cancelled());
        drop(receiver);
        assert!(sender.is_cancelled());
        // Sending after the cancel returns nothing, so there is no failure to handle here.
        sender.send(token(1));
        assert!(sender.is_cancelled());
    }

    #[test]
    fn the_channel_closes_for_the_client_when_the_engine_drops_its_sender() {
        let (sender, receiver) = egress();
        sender.send(token(1));
        drop(sender);
        assert_eq!(receiver.recv().unwrap(), token(1));
        assert!(
            receiver.recv().is_err(),
            "the stream ends after the last event"
        );
    }
}
