//! The device-facing half of the harness. Everything here type-checks GPU-free under the
//! workspace's `fallback-dynamic-loading` cudarc pin; only execution needs a driver.

pub mod alloc;
pub mod blas;
pub mod descriptor;
pub mod kernels;
pub mod runner;
pub mod step;
