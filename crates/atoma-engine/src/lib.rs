//! The executor: one pinned thread per rank, owning a device and acting on the engine thread's
//! step commands.
//!
//! The engine thread decides everything about a step on the host and hands the executor a step
//! command over a ring; the executor runs the model forward for it, samples the tokens the command
//! asks for, and hands a step result back. It re-derives nothing the command already settled.

pub mod batch;
pub mod config;
