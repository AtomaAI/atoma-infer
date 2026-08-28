//! The engine thread's seams: the step command it builds and the rings to the executor thread.

mod command;
mod rings;

pub use command::build_command;
pub use rings::{rings, EngineRings, ExecutorRings, RING_CAPACITY};
