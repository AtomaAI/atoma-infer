//! Harness tests: the ones that drive several components at once, rather than a single unit.
//!
//! `concurrency` runs on the CPU and is what a GPU-less runner exercises; `engine` and the model
//! modules need a CUDA device and are compiled behind the `cuda` feature.

pub(crate) mod concurrency;
#[cfg(feature = "cuda")]
mod engine;
pub(crate) mod fixtures;
#[cfg(all(feature = "cuda", not(feature = "nccl")))]
mod llama;
#[cfg(all(feature = "cuda", feature = "nccl"))]
mod llama_nccl;

#[cfg(feature = "cuda")]
pub fn init_tracing() {
    let _ = tracing_subscriber::fmt::try_init();
}
