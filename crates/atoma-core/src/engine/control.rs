//! Control: the bounded channel the engine thread drains before ingress on every pass.
//!
//! Control carries drain, shutdown and state queries — never cancels, which are the egress
//! receiver dropping, and never requests, which are ingress. It is a separate channel so that a
//! drain or a query cannot queue behind a burst of new requests.

use crossbeam_utils::sync::Unparker;
use flume::{Receiver, Sender, TrySendError};
use serde::{Deserialize, Serialize};

use crate::types::StepId;

/// A message to the engine thread.
#[derive(Debug)]
pub enum Control {
    /// Stop admitting waiting requests and let every request that has run finish; reply with the
    /// engine's state once nothing runs, nothing is preempted and no step is in flight.
    /// Admission does not resume. A request that never ran stays waiting and is counted in the
    /// reply; Shutdown is what finishes those.
    Drain { reply: Sender<EngineState> },
    /// Finish every live request as shut down and return from the thread.
    Shutdown,
    /// Reply with the engine's state as of this pass.
    State { reply: Sender<EngineState> },
}

/// A snapshot of the engine thread's state, answered on the thread itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineState {
    /// The step the last pass scheduled.
    pub step: StepId,
    /// Live requests in every phase; padding dummies are not counted.
    pub live_requests: usize,
    pub waiting: usize,
    pub running: usize,
    pub preempted: usize,
    /// Whether a step command is out with the executor.
    pub step_in_flight: bool,
    /// Whether admission has been stopped by a drain.
    pub draining: bool,
    pub free_blocks: usize,
    pub available_blocks: usize,
}

/// Messages the control channel holds before refusing.
pub const CONTROL_CAPACITY: usize = 16;

/// The side that sends control messages, on any thread.
#[derive(Debug, Clone)]
pub struct ControlSender {
    sender: Sender<Control>,
    wake: Unparker,
}

/// The engine thread's side of control: exactly one, on the engine thread.
#[derive(Debug)]
pub struct ControlReceiver {
    receiver: Receiver<Control>,
}

/// Opens the control channel; `wake` unparks the engine thread after every successful send.
#[must_use]
pub fn control(wake: Unparker) -> (ControlSender, ControlReceiver) {
    let (sender, receiver) = flume::bounded(CONTROL_CAPACITY);
    (ControlSender { sender, wake }, ControlReceiver { receiver })
}

impl ControlSender {
    /// Hands `control` to the engine, waking it, or refuses it.
    ///
    /// # Errors
    ///
    /// Returns the message when the channel is full or the engine thread has exited.
    pub fn try_send(&self, control: Control) -> Result<(), Control> {
        match self.sender.try_send(control) {
            Ok(()) => {
                self.wake.unpark();
                Ok(())
            }
            Err(TrySendError::Full(control) | TrySendError::Disconnected(control)) => Err(control),
        }
    }

    /// Whether the engine thread dropped its end: no control will ever be acted on again.
    #[must_use]
    pub fn engine_gone(&self) -> bool {
        self.sender.is_disconnected()
    }
}

impl ControlReceiver {
    /// The next control message, if any is waiting. Never blocks.
    #[must_use]
    pub fn try_recv(&self) -> Option<Control> {
        self.receiver.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_utils::sync::Parker;

    use super::{control, Control, EngineState, CONTROL_CAPACITY};
    use crate::types::StepId;

    #[test]
    fn control_messages_arrive_in_order_and_a_query_can_be_answered() {
        let parker = Parker::new();
        let (sender, receiver) = control(parker.unparker().clone());
        let (reply, answer) = flume::bounded(1);
        sender.try_send(Control::State { reply }).unwrap();
        sender.try_send(Control::Shutdown).unwrap();

        let Some(Control::State { reply }) = receiver.try_recv() else {
            panic!("the query was sent first");
        };
        let state = EngineState {
            step: StepId::new(3),
            live_requests: 1,
            waiting: 0,
            running: 1,
            preempted: 0,
            step_in_flight: true,
            draining: false,
            free_blocks: 4,
            available_blocks: 6,
        };
        reply.send(state).unwrap();
        assert_eq!(answer.recv().unwrap(), state);
        assert!(matches!(receiver.try_recv(), Some(Control::Shutdown)));
        assert!(receiver.try_recv().is_none());
    }

    #[test]
    fn a_full_channel_or_a_gone_engine_hands_the_message_back() {
        let parker = Parker::new();
        let (sender, receiver) = control(parker.unparker().clone());
        for _ in 0..CONTROL_CAPACITY {
            sender.try_send(Control::Shutdown).unwrap();
        }
        assert!(matches!(
            sender.try_send(Control::Shutdown),
            Err(Control::Shutdown)
        ));
        drop(receiver);
        assert!(matches!(
            sender.try_send(Control::Shutdown),
            Err(Control::Shutdown)
        ));
    }
}
