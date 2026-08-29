//! The engine thread's seam to the executor thread: the two rings that carry step commands out
//! and step results back.

mod rings;

pub use rings::{rings, EngineRings, ExecutorRings, RING_CAPACITY};
