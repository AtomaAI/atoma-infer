//! The #143 tensor-path graph spike: capture a full Llama-8B-shaped decode step with
//! `atoma-runtime`, replay it against varying inputs, and compare bit-for-bit with the same ops
//! run eagerly.
//!
//! Everything except the FlashAttention-2 FFI seam type-checks and unit-tests on a machine with
//! no GPU (the workspace's `fallback-dynamic-loading` cudarc pin compiles GPU-free), so a rig
//! session hits execution unknowns only. The `cuda` feature gates exactly the code that links the
//! vendored FA2 kernels; the `nccl` feature adds the in-graph all-reduce cells.
//!
//! This crate is deliberately throwaway: it exists to produce the findings note and the per-graph
//! memory calibration constant for the rung-2 spec, and is deleted once #143 closes.
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`dims`] | Llama-8B-shaped dimensions and derived sizes |
//! | [`matrix`] | The capture matrix: buckets × step contents, largest bucket first |
//! | [`variation`] | Deterministic per-step inputs: token ids, seq lens, block tables, slots |
//! | [`layout`] | Arena roles, static buffer sizes, device memory budget |
//! | [`splits`] | The FA2 split-KV heuristic, precomputed so accumulators allocate pre-capture |
//! | [`compare`] | Bit-identity comparison and bf16 divergence reporting |
//! | [`report`] | Findings note: capture matrix table, timings, memory measurements |
//! | [`gpu`] | Device harness: NVRTC kernels, cuBLAS, the step function, capture and replay |
//! | [`fa2`] | Paged decode attention + KV write over the FA2 FFI (`cuda` feature only) |

pub mod compare;
pub mod dims;
pub mod fa2;
pub mod gpu;
pub mod layout;
pub mod matrix;
pub mod report;
pub mod splits;
pub mod variation;
