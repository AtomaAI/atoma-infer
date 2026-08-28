//! The engine thread and its seams: ingress and control in, the heartbeat out, the step
//! command it builds, and the rings to the executor thread. The thread parks between passes and
//! is woken by any of them, or by its idle deadline.

mod command;
mod control;
mod heartbeat;
mod ingress;
#[cfg(test)]
pub(crate) mod mock;
mod rings;
#[cfg(test)]
mod tests;
mod thread;

pub use command::build_command;
pub use control::{
    control, Control, ControlReceiver, ControlSender, EngineState, CONTROL_CAPACITY,
};
pub use heartbeat::{heartbeat, Heartbeat, HeartbeatPublisher, HeartbeatReader};
pub use ingress::{ingress, IngressReceiver, IngressRefused, IngressSender};
pub use rings::{rings, EngineRings, ExecutorRings, RING_CAPACITY};
pub use thread::{Engine, EngineConfig, EngineError, EngineHandle, EngineThread, Pass};
