//! GPU-free engine core.
//!
//! Hosts the decisions the engine makes on the host side: which captured CUDA graph serves a live
//! batch, and the protocol vocabulary those decisions are written in. The crate links no driver,
//! and every test runs without a GPU.

pub mod dispatch;
pub mod kv;
pub mod protocol;
