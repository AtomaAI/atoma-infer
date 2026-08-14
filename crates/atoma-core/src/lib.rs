//! GPU-free engine core.
//!
//! Hosts the decisions the engine makes about captured CUDA graphs — which captured graph serves
//! a live batch, or why none does. The crate links no driver, and every test runs without a GPU.
