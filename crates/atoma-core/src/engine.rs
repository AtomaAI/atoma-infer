//! The engine thread's seams: the configuration it is built from, bounded ingress and control
//! in, the heartbeat out, the step command it builds, and the rings to the executor thread.

mod command;
mod config;
mod control;
mod heartbeat;
mod ingress;
mod rings;

pub use command::build_command;
pub use config::EngineConfig;
pub use control::{
    control, Control, ControlReceiver, ControlSender, EngineState, CONTROL_CAPACITY,
};
pub use heartbeat::{heartbeat, Heartbeat, HeartbeatPublisher, HeartbeatReader};
pub use ingress::{ingress, IngressReceiver, IngressRefused, IngressSender};
pub use rings::{rings, EngineRings, ExecutorRings, RING_CAPACITY};
