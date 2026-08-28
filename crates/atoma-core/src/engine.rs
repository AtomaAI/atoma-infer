//! The engine thread's seams: the rings to the executor thread.

mod rings;

pub use rings::{rings, EngineRings, ExecutorRings, RING_CAPACITY};
