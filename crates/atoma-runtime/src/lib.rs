//! CUDA graph capture substrate: context, streams, capture and graph lifetime, and the capture
//! arena.
//!
//! This crate owns device execution — the CUDA context, stream topology, graph capture, graph
//! lifetime, and the arena from which every captured step's activations are addressed. It knows
//! nothing about models, attention, or kernels; the layer whose allocation-freedom must be
//! provable stays small enough to prove.
//!
//! The crate links cudarc unconditionally under the workspace's `dynamic-loading` pin, so it
//! compiles, links, and runs `cargo test` on a machine with no CUDA toolkit, driver, or GPU.
//! Only paths that call the driver require one, and they fail loudly at the call.

pub mod arena;
pub mod capture;
pub mod context;
pub mod error;
