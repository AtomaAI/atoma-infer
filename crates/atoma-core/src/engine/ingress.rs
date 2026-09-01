//! Ingress: the bounded channel that carries requests into the engine thread.
//!
//! A refused send is the overload signal — the engine cannot take another request right now —
//! and the API's 429. Every successful send wakes the engine thread if it is parked.

use crossbeam_utils::sync::Unparker;
use flume::{Receiver, Sender, TrySendError};
use thiserror::Error;

use crate::request::NewRequest;

/// The client side of ingress: any number of these, on any thread.
#[derive(Debug, Clone)]
pub struct IngressSender {
    sender: Sender<NewRequest>,
    wake: Unparker,
}

/// The engine thread's side of ingress: exactly one, on the engine thread.
#[derive(Debug)]
pub struct IngressReceiver {
    receiver: Receiver<NewRequest>,
}

/// Why ingress refused a request. The request comes back with the refusal so the caller can
/// answer its client.
#[derive(Debug, Error)]
pub enum IngressRefused {
    /// The engine cannot take another request right now: overload.
    #[error("the engine cannot take another request right now")]
    Overload(NewRequest),
    /// The engine thread is gone; nothing will ever drain ingress again.
    #[error("the engine thread is gone")]
    EngineGone(NewRequest),
}

/// Opens ingress with room for `capacity` requests the engine has not yet drained; `wake`
/// unparks the engine thread after every successful send.
#[must_use]
pub fn ingress(capacity: usize, wake: Unparker) -> (IngressSender, IngressReceiver) {
    let (sender, receiver) = flume::bounded(capacity);
    (IngressSender { sender, wake }, IngressReceiver { receiver })
}

impl IngressSender {
    /// Hands `request` to the engine, waking it, or refuses it.
    ///
    /// # Errors
    ///
    /// Returns [`IngressRefused::Overload`] with the request when ingress is full, and
    /// [`IngressRefused::EngineGone`] when the engine thread has exited.
    pub fn try_send(&self, request: NewRequest) -> Result<(), IngressRefused> {
        match self.sender.try_send(request) {
            Ok(()) => {
                self.wake.unpark();
                Ok(())
            }
            Err(TrySendError::Full(request)) => Err(IngressRefused::Overload(request)),
            Err(TrySendError::Disconnected(request)) => Err(IngressRefused::EngineGone(request)),
        }
    }
}

impl IngressReceiver {
    /// The next request, if any is waiting. Never blocks: the engine parks on its own terms.
    #[must_use]
    pub fn try_recv(&self) -> Option<NewRequest> {
        self.receiver.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use crossbeam_utils::sync::Parker;

    use super::{ingress, IngressRefused};
    use crate::request::{
        egress, EgressReceiver, NewRequest, Priority, SamplingParams, StopCriteria,
    };
    use crate::test_support::tokens;

    fn request(clients: &mut Vec<EgressReceiver>) -> NewRequest {
        let (sender, receiver) = egress();
        clients.push(receiver);
        NewRequest {
            prompt: vec![1],
            sampling: SamplingParams::default(),
            stop: StopCriteria {
                max_new_tokens: tokens(1),
                ignore_eos: false,
            },
            priority: Priority::default(),
            egress: sender,
        }
    }

    #[test]
    fn a_full_ingress_refuses_the_request_and_hands_it_back() {
        let parker = Parker::new();
        let (sender, receiver) = ingress(2, parker.unparker().clone());
        let mut clients = Vec::new();
        sender.try_send(request(&mut clients)).unwrap();
        sender.try_send(request(&mut clients)).unwrap();

        let refused = sender.try_send(request(&mut clients)).unwrap_err();
        let IngressRefused::Overload(request) = refused else {
            panic!("a full ingress is overload, not a gone engine");
        };
        assert_eq!(request.prompt, [1]);

        assert!(receiver.try_recv().is_some(), "the engine drains in order");
        sender.try_send(request).unwrap();
        assert!(receiver.try_recv().is_some());
        assert!(receiver.try_recv().is_some());
        assert!(receiver.try_recv().is_none());
    }

    #[test]
    fn a_successful_send_wakes_a_parked_engine_thread() {
        let parker = Parker::new();
        let (sender, _receiver) = ingress(1, parker.unparker().clone());
        let engine = thread::spawn(move || parker.park());
        thread::sleep(Duration::from_millis(10));
        assert!(!engine.is_finished(), "parked until woken");

        let mut clients = Vec::new();
        sender.try_send(request(&mut clients)).unwrap();
        engine.join().unwrap();
    }

    #[test]
    fn a_gone_engine_is_reported_as_such_not_as_overload() {
        let parker = Parker::new();
        let (sender, receiver) = ingress(1, parker.unparker().clone());
        drop(receiver);
        let mut clients = Vec::new();
        assert!(matches!(
            sender.try_send(request(&mut clients)),
            Err(IngressRefused::EngineGone(_))
        ));
    }
}
