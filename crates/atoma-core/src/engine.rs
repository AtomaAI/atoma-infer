//! The engine thread's seam to the executor thread: the step command it builds from a
//! scheduling pass, and the two rings that carry commands out and results back.

mod command;
mod rings;

pub use command::build_command;
pub use rings::{rings, EngineRings, ExecutorRings, RING_CAPACITY};
